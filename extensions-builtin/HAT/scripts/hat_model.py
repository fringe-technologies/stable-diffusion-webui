import sys
import platform

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch import Tensor
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
            model = model.to(device_hat, dtype=devices.dtype).half()
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

def as_3d(img: np.ndarray) -> np.ndarray:
    """Given a grayscale image, this returns an image with 3 dimensions (image.ndim == 3)."""
    if img.ndim == 2:
        return np.expand_dims(img.copy(), axis=2)
    return img
    
def bgr_to_rgb(image: Tensor) -> Tensor:
    # flip image channels
    # https://github.com/pytorch/pytorch/issues/229
    out: Tensor = image.flip(-3)
    # RGB to BGR #may be faster:
    # out: Tensor = image[[2, 1, 0], :, :]
    return out


def rgb_to_bgr(image: Tensor) -> Tensor:
    # same operation as bgr_to_rgb(), flip image channels
    return bgr_to_rgb(image)


def bgra_to_rgba(image: Tensor) -> Tensor:
    out: Tensor = image[[2, 1, 0, 3], :, :]
    return out


def rgba_to_bgra(image: Tensor) -> Tensor:
    # same operation as bgra_to_rgba(), flip image channels
    return bgra_to_rgba(image)


def norm(x: Tensor):
    """Normalize (z-norm) from [0,1] range to [-1,1]"""
    out = (x - 0.5) * 2.0
    return out.clamp(-1, 1)
    
    
def np2tensor(
    img: np.ndarray,
    bgr2rgb=True,
    data_range=1.0,  # pylint: disable=unused-argument
    normalize=False,
    change_range=True,
    add_batch=True,
) -> Tensor:
    """Converts a numpy image array into a Tensor array.
    Parameters:
        img (numpy array): the input image numpy array
        add_batch (bool): choose if new tensor needs batch dimension added
    """

    # check how many channels the image has, then condition. ie. RGB, RGBA, Gray
    # if bgr2rgb:
    #     img = img[
    #         :, :, [2, 1, 0]
    #     ]  # BGR to RGB -> in numpy, if using OpenCV, else not needed. Only if image has colors.
    if change_range:
        dtype = img.dtype
        maxval = MAX_VALUES_BY_DTYPE.get(dtype.name, 1.0)
        t_dtype = np.dtype("float32")
        img = img.astype(t_dtype) / maxval  # ie: uint8 = /255
    # "HWC to CHW" and "numpy to tensor"
    tensor = torch.from_numpy(
        np.ascontiguousarray(np.transpose(as_3d(img), (2, 0, 1)))
    ).float()
    if bgr2rgb:
        # BGR to RGB -> in tensor, if using OpenCV, else not needed. Only if image has colors.)
        if tensor.shape[0] % 3 == 0:
            # RGB or MultixRGB (3xRGB, 5xRGB, etc. For video tensors.)
            tensor = bgr_to_rgb(tensor)
        elif tensor.shape[0] == 4:
            # RGBA
            tensor = bgra_to_rgba(tensor)
    if add_batch:
        # Add fake batch dimension = 1 . squeeze() will remove the dimensions of size 1
        tensor.unsqueeze_(0)
    if normalize:
        tensor = norm(tensor)
    return tensor


def tensor2np(
    img: Tensor,
    rgb2bgr=True,
    remove_batch=True,
    data_range=255,
    denormalize=False,
    change_range=True,
    imtype: Type = np.uint8,
) -> np.ndarray:
    """Converts a Tensor array into a numpy image array.
    Parameters:
        img (tensor): the input image tensor array
            4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
        remove_batch (bool): choose if tensor of shape BCHW needs to be squeezed
        denormalize (bool): Used to denormalize from [-1,1] range back to [0,1]
        imtype (type): the desired type of the converted numpy array (np.uint8
            default)
    Output:
        img (np array): 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    """
    n_dim = img.dim()

    # TODO: Check: could denormalize here in tensor form instead, but end result is the same

    img = img.float().cpu()

    img_np: np.ndarray

    if n_dim in (4, 3):
        # if n_dim == 4, has to convert to 3 dimensions
        if n_dim == 4 and remove_batch:
            # remove a fake batch dimension
            img = img.squeeze(dim=0)

        if img.shape[0] == 3 and rgb2bgr:  # RGB
            # RGB to BGR -> in tensor, if using OpenCV, else not needed. Only if image has colors.
            img_np = rgb_to_bgr(img).numpy()
        elif img.shape[0] == 4 and rgb2bgr:  # RGBA
            # RGBA to BGRA -> in tensor, if using OpenCV, else not needed. Only if image has colors.
            img_np = rgba_to_bgra(img).numpy()
        else:
            img_np = img.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # CHW to HWC
    elif n_dim == 2:
        img_np = img.numpy()
    else:
        raise TypeError(
            f"Only support 4D, 3D and 2D tensor. But received with dimension: {n_dim:d}"
        )

    if change_range:
        img_np = np.clip(
            data_range * img_np, 0, data_range
        ).round()  # np.clip to the data_range

    # has to be in range (0,255) before changing to np.uint8, else np.float32
    return img_np.astype(imtype)
    
    
    
def upscale_without_tiling(model, img):
    img = np.array(img)
    img_tensor = np2tensor(img, change_range=True)

    d_img = None
    try:
        d_img = img_tensor.to(device)
        d_img = d_img.half()

        result = model(d_img)
        result = tensor2np(
                result.detach().cpu().detach(),
                change_range=False,
                imtype=np.float32,
            )

        del d_img
        return Image.fromarray(output, 'RGB')
    except RuntimeError as e:
            # Check to see if its actually the CUDA out of memory error
            if "allocate" in str(e) or "CUDA" in str(e):
                # Collect garbage (clear VRAM)
                if d_img is not None:
                    try:
                        d_img.detach().cpu()
                    except:
                        pass
                    del d_img
                gc.collect()
                safe_cuda_cache_empty()
                return Split()
            else:
                # Re-raise the exception if not an OOM error
                raise            


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
