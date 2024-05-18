import logging
import sys

import torch
from PIL import Image

from modules import devices, modelloader, script_callbacks, shared, upscaler_utils
from modules.upscaler import Upscaler, UpscalerData
from modules_forge.forge_util import prepare_free_memory

SPAN_MODEL_URL = "https://huggingface.co/datasets/dputilov/TTL/resolve/main/4x-ClearRealityV1.pth"

logger = logging.getLogger(__name__)


class UpscalerSPAN(Upscaler):
    def __init__(self, dirname):
        self._cached_model = None           # keep the model when SWIN_torch_compile is on to prevent re-compile every runs
        self._cached_model_config = None    # to clear '_cached_model' when changing model (v1/v2) or settings
        self.name = "SPAN"
        self.model_url = SPAN_MODEL_URL
        self.model_name = "4x-ClearRealityV1.pth"
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

    def do_upscale(self, img: Image.Image, model_file: str) -> Image.Image:
        prepare_free_memory()

        current_config = (model_file, shared.opts.SPAN_tile)

        if self._cached_model_config == current_config:
            model = self._cached_model
        else:
            try:
                model = self.load_model(model_file)
            except Exception as e:
                print(f"Failed loading SPAN model {model_file}: {e}", file=sys.stderr)
                return img
            self._cached_model = model
            self._cached_model_config = current_config

        img = upscaler_utils.upscale_with_model(
            model,
            img,
            tile_size=shared.opts.SPAN_tile,
            tile_overlap=shared.opts.SPAN_tile_overlap,
            desc="SPAN",
        )
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

        model_descriptor = modelloader.load_spandrel_model(
            filename,
            device=self._get_device(),
            prefer_half=(devices.dtype == torch.float16),
            expected_architecture="SPAN",
        )

        return model_descriptor

    def _get_device(self):
        return devices.get_device_for('span')


def on_ui_settings():
    import gradio as gr

    shared.opts.add_option("SPAN_tile", shared.OptionInfo(192, "Tile size for all SPAN.", gr.Slider, {"minimum": 16, "maximum": 512, "step": 16}, section=('upscaling', "Upscaling")))
    shared.opts.add_option("SPAN_tile_overlap", shared.OptionInfo(32, "Tile overlap, in pixels for SwinIR. Low values = visible seam.", gr.Slider, {"minimum": 0, "maximum": 48, "step": 1}, section=('upscaling', "Upscaling")))
    
script_callbacks.on_ui_settings(on_ui_settings)
