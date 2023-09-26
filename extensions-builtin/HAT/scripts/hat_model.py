import sys
import platform

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from modules import modelloader, devices, script_callbacks, shared, images
from modules.shared import opts, state
from hat_model_arch import HAT
from modules.upscaler import Upscaler, UpscalerData

HAT_MODEL_URL = "https://huggingface.co/datasets/dputilov/TTL/resolve/main/Real_HAT_GAN_SRx4.pth"

device_hat = devices.get_device_for('hat')


class UpscalerHAT(Upscaler):
    def __init__(self, dirname):
        self._cached_model = None           # keep the model when SWIN_torch_compile is on to prevent re-compile every runs
        self._cached_model_config = None    # to clear '_cached_model' when changing model (v1/v2) or settings
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
        elif "params" in state_dict_keys:
            state_dict = state_dict["params"]
        
        model = HAT(state_dict=state_dict)

        return model


def upscale_without_tiling(model, img):
    img = np.array(img)
    img = img[:, :, ::-1]
    img = np.ascontiguousarray(np.transpose(img, (2, 0, 1))) / 255
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0).to(device_hat)
    with torch.no_grad():
        output = model(img)
    output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = 255. * np.moveaxis(output, 0, 2)
    output = output.astype(np.uint8)
    output = output[:, :, ::-1]
    return Image.fromarray(output, 'RGB')


def upscale(img, model):
    if opts.HAT_tile == 0:
        return upscale_without_tiling(model, img)

    grid = images.split_grid(img, opts.HAT_tile, opts.HAT_tile, opts.HAT_tile_overlap)
    newtiles = []
    scale_factor = 1

    for y, h, row in grid.tiles:
        newrow = []
        for tiledata in row:
            x, w, tile = tiledata

            output = upscale_without_tiling(model, tile)
            scale_factor = output.width // tile.width

            newrow.append([x * scale_factor, w * scale_factor, output])
        newtiles.append([y * scale_factor, h * scale_factor, newrow])

    newgrid = images.Grid(newtiles, grid.tile_w * scale_factor, grid.tile_h * scale_factor, grid.image_w * scale_factor, grid.image_h * scale_factor, grid.overlap * scale_factor)
    output = images.combine_grid(newgrid)
    return output


def on_ui_settings():
    import gradio as gr

    shared.opts.add_option("HAT_tile", shared.OptionInfo(192, "Tile size for all HAT.", gr.Slider, {"minimum": 16, "maximum": 512, "step": 16}, section=('upscaling', "Upscaling")))
    shared.opts.add_option("HAT_tile_overlap", shared.OptionInfo(8, "Tile overlap, in pixels for HAT. Low values = visible seam.", gr.Slider, {"minimum": 0, "maximum": 48, "step": 1}, section=('upscaling', "Upscaling")))
    if int(torch.__version__.split('.')[0]) >= 2 and platform.system() != "Windows":    # torch.compile() require pytorch 2.0 or above, and not on Windows
        shared.opts.add_option("HAT_torch_compile", shared.OptionInfo(False, "Use torch.compile to accelerate HAT.", gr.Checkbox, {"interactive": True}, section=('upscaling', "Upscaling")).info("Takes longer on first run"))


script_callbacks.on_ui_settings(on_ui_settings)
