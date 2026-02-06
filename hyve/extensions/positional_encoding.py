import torch
import torch.nn as nn
from torch.func import jacfwd, vmap

# Positional encoding as in Fourier Features paper
class PositionalEncoding(nn.Module):
	def __init__(self, size_in: int, size_out: int, scale: float) -> None:
		super().__init__()

		self.size_in = size_in
		self.size_out = size_out
		self.scale = scale

		self.register_buffer('B', torch.randn(size=(size_out, size_in)) * self.scale)
	
	def forward(self, input: torch.Tensor, *args) -> torch.Tensor:
		return torch.cat([torch.sin((2.*torch.pi*input) @ self.B.T), torch.cos((2.*torch.pi*input) @ self.B.T)], dim=-1)

	# Function for forward mode ad (tangent mode)
	def T(self, input: torch.Tensor) -> torch.Tensor:
		out = torch.cat([2*torch.pi*torch.cos((2.*torch.pi*input) @ self.B.T) @ self.B, -2*torch.pi*torch.sin((2.*torch.pi*input) @ self.B.T) @ self.B], dim=-1)
		return out


if __name__ == "__main__":
	module = PositionalEncoding(3, 6, 2).to('cuda')
	def test_fun(input):
		x = module(input)
		return torch.sum(x, dim=-1)

	# With batch dim
	input_pos = torch.rand(size=(64000, 3), device='cuda')
	input_pos.requires_grad_(True)
	# input_encoded = module(input_pos)
	# input_encoded.backward(torch.ones_like(input_encoded))
	jac = vmap(jacfwd(test_fun))(input_pos)
	# Compute forward gradient
	with torch.inference_mode():
		analytical_grad = module.T(input_pos)

	# Without batch dim
	input_pos = torch.rand(size=(2048, 3), device='cuda')
	input_encoded = module(input_pos)
	
	pass