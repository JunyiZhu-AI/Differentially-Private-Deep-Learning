# The functions for computing individual gradients are borrowed from
# https://github.com/pytorch/opacus

from typing import Union

import numpy as np
import torch
import torchvision
from torch import nn
from torch.functional import F

def _compute_conv_grad_sample(
    # for some reason pyre doesn't understand that
    # nn.Conv1d and nn.modules.conv.Conv1d is the same thing
    # pyre-ignore[11]
    layer: Union[nn.Conv2d, nn.Conv1d],
    A: torch.Tensor,
    B: torch.Tensor,
    batch_dim: int = 0,
    gpu_id=None
) -> None:
    """
    Computes per sample gradients for convolutional layers
    Args:
        layer: Layer
        A: Activations
        B: Backpropagations
        batch_dim: Batch dimension position
    """
    n = A.shape[0]
    layer_type = "Conv2d"
    # get A and B in shape depending on the Conv layer
    A = torch.nn.functional.unfold(
        A, layer.kernel_size, padding=layer.padding, stride=layer.stride
    )
    B = B.reshape(n, -1, A.shape[-1])

    # n=batch_sz; o=num_out_channels; p=(num_in_channels/groups)*kernel_sz
    grad_sample = torch.einsum("noq,npq->nop", B, A)
    # rearrange the above tensor and extract diagonals.
    grad_sample = grad_sample.view(
        n,
        layer.groups,
        -1,
        layer.groups,
        int(layer.in_channels / layer.groups),
        np.prod(layer.kernel_size),
    )
    grad_sample = torch.einsum("ngrg...->ngr...", grad_sample).contiguous()
    shape = [n] + list(layer.weight.shape)
    layer.weight.grad_sample = grad_sample.view(shape)

def _compute_linear_grad_sample(
    layer: nn.Linear, A: torch.Tensor, B: torch.Tensor, batch_dim: int = 0, gpu_id=None
) -> None:
    """
    Computes per sample gradients for ``nn.Linear`` layer
    Args:
        layer: Layer
        A: Activations
        B: Backpropagations
        batch_dim: Batch dimension position
    """
    layer.weight.grad_sample = torch.einsum("n...i,n...j->n...ij", B, A)


class CachedCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train, download, wrapper_transform, use_cache=False):
        super(CachedCIFAR10, self).__init__(root=root, train=train, transform=None, download=download)
        self.cached_data = []
        self.cached_target = []
        self.use_cache = use_cache
        self.wrapper_transform = wrapper_transform

    def __getitem__(self, index):
        if not self.use_cache:
            img, label = super(CachedCIFAR10, self).__getitem__(index)
            self.cached_data.append(img)
            self.cached_target.append(label)
        else:
            img, label = self.cached_data[index], self.cached_target[index]
        if self.wrapper_transform is not None:
            img = self.wrapper_transform(img)
        return img, label

    def set_use_cache(self, use_cache):
        if use_cache:
            self.cached_data = np.stack(self.cached_data, axis=0)
            self.cached_target = np.stack(self.cached_target, axis=0)
        else:
            self.cached_data = []
            self.cached_target = []
        self.use_cache = use_cache
