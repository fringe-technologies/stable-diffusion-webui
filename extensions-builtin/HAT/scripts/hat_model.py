import gc
import platform
import sys
from typing import Type, Tuple

import numpy as np
import torch
from PIL import Image
from hat_model_arch import HAT
from torch import Tensor
from nodes.impl.pytorch.auto_split import pytorch_auto_split
from nodes.impl.upscale.auto_split_tiles import (
    NO_TILING,
    TileSize,
    estimate_tile_size,
    parse_tile_size_input,
)
from nodes.impl.upscale.convenient_upscale import convenient_upscale
from nodes.impl.upscale.tiler import MaxTileSize
from nodes.properties.inputs import (
    BoolInput,
    ImageInput,
    SrModelInput,
    TileSizeDropdown,
)
from nodes.properties.outputs import ImageOutput
from nodes.utils.utils import get_h_w_c

from modules import modelloader, devices, script_callbacks, shared, images
from modules.shared import opts
from modules.upscaler import Upscaler, UpscalerData

HAT_MODEL_URL = "https://huggingface.co/datasets/dputilov/TTL/resolve/main/Real_HAT_GAN_SRx4.pth"

device_hat = devices.get_device_for('hat')

class UpscalerHAT(Upscaler):
    def __init__(self, dirname):
        self._cached_model = None
        self._cached_model_config = None
        self.name = "HAT"
        self.model_url = HAT_MODEL_URL
        self.model_name = "HAT 4x"
        self.user_path = dirname
        super().__init__()
        scalers = []
        model_files = self.find_models(ext_filter=[".pt", ".pth"])
        for model in model_files:
            if model.startswith("http"):
                name = self.model_name
            else:
                name = modelloader.friendly_name(model)
            model_data = UpscalerData(name, model, self)
            scalers.append(model_data)
        self.scalers = scalers

    def do_upscale(self, img, model_file):
        use_compile = hasattr(opts, 'HAT_torch_compile') and opts.HAT_torch_compile \
                      and int(torch.__version__.split('.')[0]) >= 2 and platform.system() != "Windows"
        current_config = (model_file, opts.HAT_tile)

        if use_compile and self._cached_model_config == current_config:
            model = self._cached_model
        else:
            self._cached_model = None
            try:
                model = self.load_model(model_file)
            except Exception as e:
                print(f"Failed loading HAT model {model_file}: {e}", file=sys.stderr)
                return img
            model = model.to(device_hat, dtype=devices.dtype)
            if use_compile:
                model = torch.compile(model)
                self._cached_model = model
                self._cached_model_config = current_config
        img = upscale(img, model)
        devices.torch_gc()
        return img

    def load_model(self, path, scale=4):
        if path.startswith("http"):
            filename = modelloader.load_file_from_url(
                url=path,
                model_dir=self.model_download_path,
                file_name=f"{self.model_name.replace(' ', '_')}.pth",
            )
        else:
            filename = path

        state_dict = torch.load(filename)

        state_dict_keys = list(state_dict.keys())

        if "params_ema" in state_dict_keys:
            state_dict = state_dict["params_ema"]
        elif "params-ema" in state_dict_keys:
            state_dict = state_dict["params-ema"]

        model = HAT(state_dict=state_dict,
                    upscale=4,
                    in_chans=3,
                    img_size=64,
                    window_size=16,
                    compress_ratio=3,
                    squeeze_factor=30,
                    conv_scale=0.01,
                    overlap_ratio=0.5,
                    img_range=1.,
                    depths=[6, 6, 6, 6, 6, 6],
                    embed_dim=180,
                    num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2,
                    upsampler='pixelshuffle',
                    resi_connection='1conv')

        return model


def upscale(
    img,
    model
):
    with torch.no_grad():
        use_fp16 = False
        device = 'cuda'

        def estimate():
            if "cuda" in device.type:
                mem_info: Tuple[int, int] = torch.cuda.mem_get_info(device)
                free, _total = mem_info
                element_size = 2 if use_fp16 else 4
                model_bytes = sum(p.numel() * element_size for p in model.parameters())
                budget = int(free * 0.8)

                return MaxTileSize(
                    estimate_tile_size(
                        budget,
                        model_bytes,
                        img,
                        element_size,
                    )
                )
            return MaxTileSize()

        # Disable tiling for SCUNet
        upscale_tile_size = shared.opts.HAT_tile

        img_out = pytorch_auto_split(
            img,
            model=model,
            device=device,
            use_fp16=use_fp16,
            tiler=parse_tile_size_input(upscale_tile_size, estimate),
        )

        return Image.fromarray(img_out).convert('RGB')


def on_ui_settings():
    import gradio as gr

    shared.opts.add_option("HAT_tile", shared.OptionInfo(192, "Tile size for all HAT.", gr.Slider,
                                                         {"minimum": 16, "maximum": 512, "step": 16},
                                                         section=('upscaling', "Upscaling")))
    shared.opts.add_option("HAT_tile_overlap",
                           shared.OptionInfo(8, "Tile overlap, in pixels for HAT. Low values = visible seam.",
                                             gr.Slider, {"minimum": 0, "maximum": 48, "step": 1},
                                             section=('upscaling', "Upscaling")))
    if int(torch.__version__.split('.')[
               0]) >= 2 and platform.system() != "Windows":  # torch.compile() require pytorch 2.0 or above, and not on Windows
        shared.opts.add_option("HAT_torch_compile",
                               shared.OptionInfo(False, "Use torch.compile to accelerate HAT.", gr.Checkbox,
                                                 {"interactive": True}, section=('upscaling', "Upscaling")).info(
                                   "Takes longer on first run"))


script_callbacks.on_ui_settings(on_ui_settings)
