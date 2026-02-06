import torch.nn as nn
import torch
import numpy as np
from typing import Optional
from einops import unpack
from .hyve import Sine

class Decoder(nn.Module):
    latent_size: int
    hidden_size: int

class FCDecoder(Decoder):
	def __init__(self, latent_size: int = 64, hidden_size: int = 64, num_hidden_layers: int = 2, activation: nn.Module = Sine(30.), modulation_activation: Optional[nn.Module] = nn.ReLU(), use_first_layer_init: bool = False,*args, **kwargs) -> None: 
		super().__init__(*args, **kwargs)

		self.latent_size = latent_size
		self.hidden_size = hidden_size
		self.num_hidden_layers = num_hidden_layers
		self.activation = activation
		self.modulation_activation = modulation_activation
		self.use_first_layer_init = use_first_layer_init

		self.layers = nn.ModuleList()
		self.modulation = nn.ModuleList() if self.modulation_activation is not None else None

		# Add first layer
		if self.modulation is not None and self.modulation_activation is not None:
			self.layers.append(nn.Sequential(
				nn.Conv1d(3, self.hidden_size, 1), 
				self.activation
			))
			self.modulation.append(nn.Sequential(
				nn.Conv1d(self.latent_size, self.hidden_size, 1),
				self.modulation_activation
			))
		else:
			self.layers.append(nn.Sequential(
				nn.Conv1d(self.latent_size+3, self.hidden_size, 1), 
				self.activation
			))
		
		# Add main layers
		for _ in range(self.num_hidden_layers):
			self.layers.append(nn.Sequential(
				nn.Conv1d(self.hidden_size, self.hidden_size, 1), 
				self.activation
			))

			if self.modulation is not None and self.modulation_activation is not None:
				self.modulation.append(nn.Sequential(
					nn.Conv1d(self.hidden_size, self.hidden_size, 1),
					self.modulation_activation
				))

		# Add last layer
		self.layers.append(nn.Conv1d(self.hidden_size, 1, 1))

		self._init_weights()

	
	@torch.no_grad()
	def _init_weights(self):
		if isinstance(self.activation, Sine):
			omega_0 = self.activation.omega_0
			for mod in self.modules():
				if isinstance(mod, nn.Conv1d):
					mod.weight.uniform_(
						-np.sqrt(6./mod.in_channels) / omega_0, 
						 np.sqrt(6./mod.in_channels) / omega_0)
			if self.use_first_layer_init:
				first_layer = self.layers[0]
				if isinstance(first_layer, nn.Sequential):
					mod = first_layer[0]
					if isinstance(mod, nn.Conv1d):
						mod.weight.uniform_(
							-1/(mod.in_channels),
							1/(mod.in_channels)
						)


	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		x: Tensor (b (n_latent 3) n_points)
		Assume that x has been formatted, such that this decoder can simply be run on all instances equally.
		"""

		if self.modulation is not None:
			x, sd = unpack(x, [[self.latent_size], [3]], "b * n")
			x_skip = 0
			for i, layer in enumerate(self.modulation):
				x = layer(x) + x_skip
				x_skip = x
				sd = self.layers[i](sd) * x
			sd = self.layers[-1](sd)
		else:
			sd = x
			for layer in self.layers:
				sd = layer(sd)

		return sd