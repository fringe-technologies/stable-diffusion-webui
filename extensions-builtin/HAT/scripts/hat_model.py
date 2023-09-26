import sys
import platform

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from modules import modelloader, devices, script_callbacks, shared
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
        
        print(state_dict_keys)
        model = HAT(
                state_dict=state_dict,
                upscale=scale,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.0
            )

        return model


def upscale(
        img,
        model,
        tile=None,
        tile_overlap=None,
        window_size=8,
        scale=4,
):
    tile = tile or opts.HAT_tile
    tile_overlap = tile_overlap or opts.HAT_tile_overlap


    img = np.array(img)
    img = img[:, :, ::-1]
    img = np.moveaxis(img, 2, 0) / 255
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0).to(device_hat, dtype=devices.dtype)
    with torch.no_grad(), devices.autocast():
        _, _, h_old, w_old = img.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, : h_old + h_pad, :]
        img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, : w_old + w_pad]
        output = inference(img, model, tile, tile_overlap, window_size, scale)
        output = output[..., : h_old * scale, : w_old * scale]
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(
                output[[2, 1, 0], :, :], (1, 2, 0)
            )  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        return Image.fromarray(output, "RGB")


def inference(img, model, tile, tile_overlap, window_size, scale):
    # test the image tile by tile
    b, c, h, w = img.size()
    tile = min(tile, h, w)
    assert tile % window_size == 0, "tile size should be a multiple of window_size"
    sf = scale

    stride = tile - tile_overlap
    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
    E = torch.zeros(b, c, h * sf, w * sf, dtype=devices.dtype, device=device_hat).type_as(img)
    W = torch.zeros_like(E, dtype=devices.dtype, device=device_hat)

    with tqdm(total=len(h_idx_list) * len(w_idx_list), desc="HAT tiles") as pbar:
        for h_idx in h_idx_list:
            if state.interrupted or state.skipped:
                break

            for w_idx in w_idx_list:
                if state.interrupted or state.skipped:
                    break

                in_patch = img[..., h_idx: h_idx + tile, w_idx: w_idx + tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[
                ..., h_idx * sf: (h_idx + tile) * sf, w_idx * sf: (w_idx + tile) * sf
                ].add_(out_patch)
                W[
                ..., h_idx * sf: (h_idx + tile) * sf, w_idx * sf: (w_idx + tile) * sf
                ].add_(out_patch_mask)
                pbar.update(1)
    output = E.div_(W)

    return output


def on_ui_settings():
    import gradio as gr

    shared.opts.add_option("HAT_tile", shared.OptionInfo(192, "Tile size for all HAT.", gr.Slider, {"minimum": 16, "maximum": 512, "step": 16}, section=('upscaling', "Upscaling")))
    shared.opts.add_option("HAT_tile_overlap", shared.OptionInfo(8, "Tile overlap, in pixels for HAT. Low values = visible seam.", gr.Slider, {"minimum": 0, "maximum": 48, "step": 1}, section=('upscaling', "Upscaling")))
    if int(torch.__version__.split('.')[0]) >= 2 and platform.system() != "Windows":    # torch.compile() require pytorch 2.0 or above, and not on Windows
        shared.opts.add_option("HAT_torch_compile", shared.OptionInfo(False, "Use torch.compile to accelerate HAT.", gr.Checkbox, {"interactive": True}, section=('upscaling', "Upscaling")).info("Takes longer on first run"))


script_callbacks.on_ui_settings(on_ui_settings)
