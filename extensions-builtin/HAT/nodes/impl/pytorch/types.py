from typing import Union

from .architecture.HAT import HAT

PyTorchSRModels = (
    HAT
)
PyTorchSRModel = Union[
    HAT
]


def is_pytorch_sr_model(model: object):
    return isinstance(model, PyTorchSRModels)


PyTorchModels = PyTorchSRModels
PyTorchModel = Union[PyTorchSRModel]


def is_pytorch_model(model: object):
    return isinstance(model, PyTorchModels)
