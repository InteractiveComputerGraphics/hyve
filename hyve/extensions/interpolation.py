import torch
from .interpolation_cuda.interpolation import LinearInterpolationFunction, PointToGridFunction

from typing import Any, Callable, Optional

def uniform_grid_to_points_linear(features: torch.Tensor, samples_flat: torch.Tensor, samples_batch: torch.Tensor, ngrid: int) -> Any:
    return LinearInterpolationFunction.apply(features, samples_flat, samples_batch, ngrid)

def points_to_uniform_grid_linear(sample_features: torch.Tensor, samples_flat: torch.Tensor, samples_batch: torch.Tensor, ngrid: int, min_weight: Optional[float] = None, mask: Optional[torch.Tensor] = None, mask_value: Optional[float] = None, slope: Optional[float] = None, log_grad_fn: Optional[Callable[...,Any]]=None) -> Any:
    features, counts, weights = PointToGridFunction.apply(sample_features, samples_flat, samples_batch, ngrid, min_weight, mask, mask_value, slope, log_grad_fn)
    return features/counts.clamp(min=min_weight if min_weight is not None else 1e-6), weights

def points_to_uniform_grid_linear_unnormalized(sample_features: torch.Tensor, samples_flat: torch.Tensor, samples_batch: torch.Tensor, ngrid: int, min_weight: Optional[float] = None, mask: Optional[torch.Tensor] = None) -> Any:
    features, _ = PointToGridFunction.apply(sample_features, samples_flat, samples_batch, ngrid, min_weight, mask)
    return features
