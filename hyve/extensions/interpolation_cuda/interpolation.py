from typing import Any, Callable, Optional
import torch
import os
from torch.autograd import Function, gradcheck, gradgradcheck
from torch.nn import Module
from torch.utils.cpp_extension import load
from einops import pack

# Load the linear interpolation CUDA extension module
try:
	import linear_interpolation_cuda
except ImportError:
	local_source = ["linear_interpolation_cuda.cpp", "linear_interpolation_cuda_kernel.cu"]
	cpp_source = [os.path.join(os.path.dirname(os.path.realpath(__file__)), file) for file in local_source]
	linear_interpolation_cuda = load(name="linear_interpolation_cuda", sources=cpp_source, verbose=True)# , extra_cuda_cflags=['-DTORCH_USE_CUDA_DSA'])

# Load the point to grid CUDA extension module
try:
	import point_to_grid_cuda
except ImportError:
	local_source = ["point_to_grid_cuda.cpp", "point_to_grid_cuda_kernel.cu"]
	cpp_source = [os.path.join(os.path.dirname(os.path.realpath(__file__)), file) for file in local_source]
	point_to_grid_cuda = load(name="point_to_grid_cuda", sources=cpp_source, verbose=True,)

class LinearInterpolationFunction(Function):
	@staticmethod
	def forward(ctx, features, samples_flat, samples_batch, ngrid):
		with torch.no_grad():
			i_z, _, w, cluster_indices = linear_interpolation_cuda.forward(features.contiguous(), samples_flat.contiguous(), samples_batch, ngrid)
			ctx.save_for_backward(samples_flat, w, cluster_indices, features)
			ctx.ngrid = ngrid
		return i_z

	@staticmethod
	def backward(ctx, grad_output):
		samples_flat, w, cluster_indices, features = ctx.saved_tensors
		ngrid = ctx.ngrid
		grad_features, grad_samples_flat = LinearInterpolationBackwardFunction.apply(grad_output, samples_flat.contiguous(), features.contiguous(), w, cluster_indices, ngrid)
		# grad_features, grad_samples_flat = linear_interpolation_cuda.backward(grad_output, i_000, local_coord, w, cluster_indices)
		return grad_features, grad_samples_flat, None, None


class LinearInterpolationBackwardFunction(Function):
	@staticmethod
	def forward(ctx, grad_output, samples_flat, features, w, cluster_indices, ngrid):
		with torch.no_grad():
			grad_features, grad_samples_flat, grad_w = linear_interpolation_cuda.backward(grad_output.contiguous(), samples_flat.contiguous(), features, w, cluster_indices, ngrid)
			ctx.save_for_backward(grad_output, samples_flat, w, grad_w, cluster_indices, features)
			ctx.ngrid = ngrid
		return grad_features, grad_samples_flat

	@staticmethod
	def backward(ctx, grad_grad_features, grad_grad_samples_flat):
		grad_output, samples_flat, w, grad_w, cluster_indices, features = ctx.saved_tensors
		grad_grad_output, grad_features, grad_samples_flat = linear_interpolation_cuda.double_backward(grad_grad_features, grad_grad_samples_flat, grad_output.contiguous(), samples_flat.contiguous(), features, w, grad_w, cluster_indices, ctx.ngrid)
		return grad_grad_output, grad_features, grad_samples_flat, None, None, None

class PointToGridFunction(Function):
	@staticmethod
	def forward(ctx, samples_features, samples_flat, samples_batch, ngrid, min_weight: Optional[float]=None, mask: Optional[torch.Tensor]=None, mask_value: Optional[float] = None, slope: Optional[float] = None, log_grad_fn: Optional[Callable[...,Any]]=None):
		with torch.no_grad():
			if min_weight is None and mask is None and mask_value is None and slope is None:
				i_z, counts, _, w, cluster_indices = point_to_grid_cuda.forward(samples_features.contiguous(), samples_flat.contiguous(), samples_batch, ngrid)
			elif min_weight is not None and mask is None and mask_value is None and slope is None:
				i_z, counts, _, w, cluster_indices = point_to_grid_cuda.forward_minw(samples_features.contiguous(), samples_flat.contiguous(), samples_batch, ngrid, min_weight)
			elif min_weight is None and mask is not None and mask_value is None and slope is None:
				i_z, counts, _, w, cluster_indices = point_to_grid_cuda.forward_masked(samples_features.contiguous(), samples_flat.contiguous(), samples_batch, ngrid, mask)
			elif min_weight is None and mask is None and mask_value is not None and slope is None:
				i_z, counts, _, w, cluster_indices, _mask = point_to_grid_cuda.forward_masked_weighted(samples_features.contiguous(), samples_flat.contiguous(), samples_batch, ngrid, mask_value)
			elif min_weight is None and mask is None and mask_value is None and slope is not None:
				i_z, counts, _, w, cluster_indices = point_to_grid_cuda.forward_sloped(samples_features.contiguous(), samples_flat.contiguous(), samples_batch, ngrid, slope)
			else:
				raise ValueError("Only one of min_weight, mask or mask_value can be provided")
			if mask_value is not None:
				ctx.save_for_backward(samples_flat, w, cluster_indices, samples_features, _mask)
			else:
				ctx.save_for_backward(samples_flat, w, cluster_indices, samples_features)
			ctx.ngrid = ngrid
			ctx.min_weight = min_weight
			ctx.mask = mask
			ctx.log_grad_fn = log_grad_fn
			ctx.mask_value = mask_value
			ctx.slope = slope
		return i_z, counts, w

	@staticmethod
	def backward(ctx, grad_output, grad_counts, grad_w):
		ngrid = ctx.ngrid
		min_weight = ctx.min_weight
		mask = ctx.mask
		mask_value = ctx.mask_value
		slope = ctx.slope
		if mask_value is not None:
			samples_flat, w, cluster_indices, features, _mask = ctx.saved_tensors
		else:
			samples_flat, w, cluster_indices, features = ctx.saved_tensors
		if min_weight is None and mask is None and mask_value is None and slope is None:
			grad_samples_features, grad_samples_flat = point_to_grid_cuda.backward(grad_output, grad_counts, samples_flat.contiguous(), features.contiguous(), w, cluster_indices, ngrid)
		elif min_weight is not None and mask is None and mask_value is None and slope is None:
			grad_samples_features, grad_samples_flat = point_to_grid_cuda.backward_minw(grad_output, grad_counts, samples_flat.contiguous(), features.contiguous(), w, cluster_indices, ngrid, min_weight)
		elif min_weight is None and mask is not None and mask_value is None and slope is None:
			grad_samples_features, grad_samples_flat = point_to_grid_cuda.backward_masked(grad_output, grad_counts, samples_flat.contiguous(), features.contiguous(), w, cluster_indices, ngrid, mask)
		elif min_weight is None and mask is None and mask_value is not None and slope is None:
			grad_samples_features, grad_samples_flat = point_to_grid_cuda.backward_masked_weighted(grad_output, grad_counts, samples_flat.contiguous(), features.contiguous(), w, cluster_indices, ngrid, mask_value, _mask)
		elif min_weight is None and mask is None and mask_value is None and slope is not None:
			grad_samples_features, grad_samples_flat = point_to_grid_cuda.backward_sloped(grad_output, grad_counts, samples_flat.contiguous(), features.contiguous(), w, cluster_indices, ngrid, slope)
		
		if ctx.log_grad_fn is not None:
			ctx.log_grad_fn({
				f"grad_f_p2g_{ngrid}": grad_samples_features.detach().norm(),
				f"grad_s_p2g_{ngrid}": grad_samples_flat.detach().norm()
			})

		return grad_samples_features, grad_samples_flat, None, None, None, None, None, None, None

class LinearInterpolationModule(Module):
	def __init__(self):
		super(LinearInterpolationModule, self).__init__()

	def forward(self, features, samples_flat, samples_batch, ngrid):
		return LinearInterpolationFunction.apply(features, samples_flat, samples_batch, ngrid)

class PointToGridModule(Module):
	def __init__(self):
		super(PointToGridModule, self).__init__()

	def forward(self, samples_features, samples_flat, samples_batch, ngrid, min_weight=None, mask=None, mask_value=None, slope=None, log_grad_fn=None):
		features, weights = PointToGridFunction.apply(samples_features, samples_flat, samples_batch, ngrid, min_weight, mask, mask_value, slope, log_grad_fn)
		return features/weights.clamp(min=min_weight if min_weight is not None else 1e-6)

def time_cuda_function(cuda_function_lambda):
	# Create CUDA events
	start_event = torch.cuda.Event(enable_timing=True)
	end_event = torch.cuda.Event(enable_timing=True)

	# Record the start event
	start_event.record()

	# Execute the CUDA function
	# with torch.no_grad():
	result = cuda_function_lambda()

	# Record the end event
	end_event.record()

	# Synchronize to ensure all CUDA operations are complete
	torch.cuda.synchronize()

	# Calculate the elapsed time
	elapsed_time_ms = start_event.elapsed_time(end_event)

	print(f"Elapsed Time: {elapsed_time_ms:.4f} ms")

	return result, elapsed_time_ms

if __name__ == "__main__":
	from hyve.extensions.interpolation import linear_interpolation
	from torch.autograd.functional import jacobian, hessian
	# Usage
	linear_interp_module = LinearInterpolationModule()
	point_to_grid_module = PointToGridModule()

	# Move inputs to GPU if needed
	ngrid = 2
	features = torch.rand((ngrid*ngrid*ngrid, 3)).double().to('cuda:0')
	features.requires_grad_(True)
	# grid_features = torch.stack(torch.meshgrid([torch.linspace(-1, 1, ngrid) for _ in range(3)]), dim=-1).view(-1, 3).double().to('cuda:0')
	grid_features = torch.tensor([[-1, -1, -1],
						 [1, -1, -1],
						 [-1, 1, -1],
						 [1, 1, -1],
						 [-1, -1, 1],
						 [1, -1, 1],
						 [-1, 1, 1],
						 [1, 1, 1],
						 ]).double().reshape(8, 3).to('cuda:0')
	# grid_features, _ = pack([grid_features, grid_features], "* coord")
	# ngrid = 2
	# grid_features = features
	grid_features.requires_grad_(True)

	samples_flat = torch.rand((1, 3)).double().to('cuda:0')
	# samples_flat, _ = pack([samples_flat/2 - 0.5, samples_flat/2 + 0.5], "* coord")
	samples_flat.requires_grad_(True)

	samples_batch = torch.zeros((samples_flat.shape[0]), dtype=torch.int64).to('cuda:0')
	# samples_batch[1:] = 1
	# samples_batch[70:] = 2
	# samples_batch_1 = torch.ones((1), dtype=torch.int64).to('cuda:0')
	# samples_batch, _ = pack([samples_batch_0, samples_batch_0], "*")

	def test_linear_interpolation():
		# Forward pass
		i_z_cuda, time_cuda = time_cuda_function(lambda: linear_interp_module(grid_features, samples_flat, samples_batch, ngrid))
		i_z_cuda = linear_interp_module(grid_features, samples_flat, samples_batch, ngrid)
		i_z_python, time_python = time_cuda_function(lambda: linear_interpolation(grid_features, samples_flat, samples_batch, ngrid)[0])
		# i_z = linear_interp_module(features, samples_flat, samples_batch, ngrid)

		is_interpolation_correct = torch.allclose(i_z_cuda, samples_flat)
		# is_interpolation_correct = False

		is_cuda_correct = torch.allclose(i_z_cuda, i_z_python)

		print(f"Cuda interpolation correct: {is_interpolation_correct}")
		print(f"Cuda matches python: {is_cuda_correct}")
		print(f"Cuda speedup: {time_python/time_cuda}")

		
		fun = lambda *x: LinearInterpolationFunction.apply(*x, samples_batch, ngrid)
		# def python_fun(*args):
		# 	return linear_interpolation(*args, samples_batch, ngrid)[0]
		# def python_fun_scalar(*args):
		# 	return linear_interpolation(*args, samples_batch, ngrid)[0].mean()
		is_jac_correct_cuda, time_gradcheck_cuda = time_cuda_function(lambda: gradcheck(fun, [grid_features, samples_flat]))
		is_hess_correct_cuda, time_gradgradcheck_cuda = time_cuda_function(lambda: gradgradcheck(fun, [grid_features, samples_flat]))
		jac_cuda, time_jac_cuda = time_cuda_function(lambda: jacobian(fun, (grid_features.float(), samples_flat.float())))
		# hess_cuda, time_hess_cuda = time_cuda_function(lambda: hessian(fun_scalar, (grid_features.float(), samples_flat.float())))
		# is_jac_correct_python, time_gradcheck_python = time_cuda_function(lambda: gradcheck(python_fun, [grid_features, samples_flat]))
		# is_hess_correct_python, time_gradgradcheck_python = time_cuda_function(lambda: gradgradcheck(python_fun, [grid_features, samples_flat]))
		# jac_python, time_jac_python = time_cuda_function(lambda: jacobian(python_fun, (grid_features.float(), samples_flat.float())))
		# hess_python, time_hess_python = time_cuda_function(lambda: hessian(python_fun_scalar, (grid_features.float(), samples_flat.float())))

		# cuda_jac_matches_python = torch.allclose(jac_cuda[0], jac_python[0]) and torch.allclose(jac_cuda[1], jac_python[1])

		print(f"Gradient is correct cuda: {is_jac_correct_cuda}")
		print(f"GradGrad is correct cuda: {is_hess_correct_cuda}")
		# print(f"Gradient is correct python: {is_jac_correct_python}")
		# print(f"GradGrad is correct python: {is_hess_correct_python}")
		# print(f"Cuda Jacobian matches python Jacobian: {cuda_jac_matches_python}")
		# print(f"Speedup gradient computation: {time_jac_python/time_jac_cuda}")
		# print(f"Speedup gradgrad computation: {time_gradgradcheck_python/time_gradgradcheck_cuda}")

	def test_point_to_grid():
		ngrid=16
		samples = 30
		samples_flat_0 = torch.rand((samples, 3)).double().to('cuda:0') * 2 - 1
		samples_flat_1 = torch.rand((samples, 3)).double().to('cuda:0') * 2 - 1
		# samples_flat, _ = pack([samples_flat/2 - 0.5, samples_flat/2 + 0.5], "* coord")
		samples_batch_0 = torch.zeros((samples), dtype=torch.int64).to('cuda:0')
		samples_batch_1 = torch.ones((samples), dtype=torch.int64).to('cuda:0')

		samples_flat, _ = pack([samples_flat_0, samples_flat_1], "* coord")
		samples_flat.requires_grad_(True)
		samples_batch, _ = pack([samples_batch_0, samples_batch_1], "*")


		# Forward pass
		i_z_cuda, time_cuda = time_cuda_function(lambda: point_to_grid_module(samples_flat, samples_flat, samples_batch, ngrid))
		print(f"Time for forward pass: {time_cuda}")
		print(f"Batching works correctly: {not torch.allclose(i_z_cuda[:ngrid**3], i_z_cuda[ngrid**3:])}")

		# Interpolate back
		sample_features = LinearInterpolationFunction.apply(i_z_cuda, samples_flat, samples_batch, ngrid)
		print(f"Average distance and std to original features: {(sample_features - samples_flat).norm(dim=1).mean()}, {(sample_features - samples_flat).norm(dim=1).std()}")

		# Check gradient		
		fun = lambda *x: PointToGridFunction.apply(*x, samples_batch, ngrid)
		is_jac_correct_cuda, time_gradcheck_cuda = time_cuda_function(lambda: gradcheck(fun, [samples_flat, samples_flat]))

		print(f"Gradient is correct cuda: {is_jac_correct_cuda}")
		pass

	def test_point_to_grid_min_weight():
		ngrid=4
		samples = 8
		samples_flat_0 = torch.rand((samples, 3)).double().to('cuda:0') * 2 - 1
		samples_flat_1 = torch.rand((samples, 3)).double().to('cuda:0') * 2 - 1
		# samples_flat, _ = pack([samples_flat/2 - 0.5, samples_flat/2 + 0.5], "* coord")
		samples_batch_0 = torch.zeros((samples), dtype=torch.int64).to('cuda:0')
		samples_batch_1 = torch.ones((samples), dtype=torch.int64).to('cuda:0')

		samples_flat, _ = pack([samples_flat_0, samples_flat_1], "* coord")
		samples_flat.requires_grad_(True)
		samples_batch, _ = pack([samples_batch_0, samples_batch_1], "*")


		# Forward pass
		i_z_cuda, time_cuda = time_cuda_function(lambda: point_to_grid_module(samples_flat, samples_flat, samples_batch, ngrid, min_weight=0.05))
		print(f"Time for forward pass: {time_cuda}")
		print(f"Batching works correctly: {not torch.allclose(i_z_cuda[:ngrid**3], i_z_cuda[ngrid**3:])}")

		# Interpolate back
		sample_features = LinearInterpolationFunction.apply(i_z_cuda, samples_flat, samples_batch, ngrid)
		print(f"Average distance and std to original features: {(sample_features - samples_flat).norm(dim=1).mean()}, {(sample_features - samples_flat).norm(dim=1).std()}")

		# Check gradient		
		fun = lambda *x: PointToGridFunction.apply(*x, samples_batch, ngrid, 0.1)
		is_jac_correct_cuda, time_gradcheck_cuda = time_cuda_function(lambda: gradcheck(fun, [samples_flat, samples_flat]))

		print(f"Gradient is correct cuda: {is_jac_correct_cuda}")
		pass

	def test_point_to_grid_masked():
		ngrid=16
		samples = 30
		samples_flat_0 = torch.rand((samples, 3)).double().to('cuda:0') * 2 - 1
		samples_flat_1 = torch.rand((samples, 3)).double().to('cuda:0') * 2 - 1
		# samples_flat, _ = pack([samples_flat/2 - 0.5, samples_flat/2 + 0.5], "* coord")
		samples_batch_0 = torch.zeros((samples), dtype=torch.int64).to('cuda:0')
		samples_batch_1 = torch.ones((samples), dtype=torch.int64).to('cuda:0')

		samples_flat, _ = pack([samples_flat_0, samples_flat_1], "* coord")
		samples_flat.requires_grad_(True)
		samples_batch, _ = pack([samples_batch_0, samples_batch_1], "*")


		# Forward pass
		mask = torch.rand((samples_flat.shape[0], 8), device='cuda:0')
		mask[torch.arange(mask.shape[0])[:, None], mask.topk(dim=1,k=2).indices] = 0
		mask = mask <= 0
		i_z_cuda, time_cuda = time_cuda_function(lambda: point_to_grid_module(samples_flat, samples_flat, samples_batch, ngrid, mask=mask))
		print(f"Time for forward pass: {time_cuda}")
		print(f"Batching works correctly: {not torch.allclose(i_z_cuda[:ngrid**3], i_z_cuda[ngrid**3:])}")

		# Interpolate back
		sample_features = LinearInterpolationFunction.apply(i_z_cuda, samples_flat, samples_batch, ngrid)
		print(f"Average distance and std to original features: {(sample_features - samples_flat).norm(dim=1).mean()}, {(sample_features - samples_flat).norm(dim=1).std()}")

		# Check gradient		
		fun = lambda *x: PointToGridFunction.apply(*x, samples_batch, ngrid, None, mask)
		is_jac_correct_cuda, time_gradcheck_cuda = time_cuda_function(lambda: gradcheck(fun, [samples_flat, samples_flat]))

		print(f"Gradient is correct cuda: {is_jac_correct_cuda}")
		pass


	def test_point_to_grid_masked_weighted():
		ngrid=16
		samples = 30
		samples_flat_0 = torch.rand((samples, 3)).double().to('cuda:0') * 2 - 1
		samples_flat_1 = torch.rand((samples, 3)).double().to('cuda:0') * 2 - 1
		# samples_flat, _ = pack([samples_flat/2 - 0.5, samples_flat/2 + 0.5], "* coord")
		samples_batch_0 = torch.zeros((samples), dtype=torch.int64).to('cuda:0')
		samples_batch_1 = torch.ones((samples), dtype=torch.int64).to('cuda:0')

		samples_flat, _ = pack([samples_flat_0, samples_flat_1], "* coord")
		samples_flat.requires_grad_(True)
		samples_batch, _ = pack([samples_batch_0, samples_batch_1], "*")


		# Forward pass
		mask_value = 0.5
		i_z_cuda, time_cuda = time_cuda_function(lambda: point_to_grid_module(samples_flat, samples_flat, samples_batch, ngrid, mask_value=mask_value))
		print(f"Time for forward pass: {time_cuda}")
		print(f"Batching works correctly: {not torch.allclose(i_z_cuda[:ngrid**3], i_z_cuda[ngrid**3:])}")

		# Interpolate back
		sample_features = LinearInterpolationFunction.apply(i_z_cuda, samples_flat, samples_batch, ngrid)
		print(f"Average distance and std to original features: {(sample_features - samples_flat).norm(dim=1).mean()}, {(sample_features - samples_flat).norm(dim=1).std()}")

		# Check gradient		
		def fun(*x):
			torch.manual_seed(0)
			return PointToGridFunction.apply(*x, samples_batch, ngrid, None, None, mask_value)
		is_jac_correct_cuda, time_gradcheck_cuda = time_cuda_function(lambda: gradcheck(fun, [samples_flat, samples_flat]))

		print(f"Gradient is correct cuda: {is_jac_correct_cuda}")
		pass

	def test_point_to_grid_sloped():
		ngrid=16
		samples = 30
		samples_flat_0 = torch.rand((samples, 3)).double().to('cuda:0') * 2 - 1
		samples_flat_1 = torch.rand((samples, 3)).double().to('cuda:0') * 2 - 1
		# samples_flat, _ = pack([samples_flat/2 - 0.5, samples_flat/2 + 0.5], "* coord")
		samples_batch_0 = torch.zeros((samples), dtype=torch.int64).to('cuda:0')
		samples_batch_1 = torch.ones((samples), dtype=torch.int64).to('cuda:0')

		samples_flat, _ = pack([samples_flat_0, samples_flat_1], "* coord")
		samples_flat.requires_grad_(True)
		samples_batch, _ = pack([samples_batch_0, samples_batch_1], "*")


		# Forward pass
		slope = 0
		i_z_cuda_gt, time_cuda_gt = time_cuda_function(lambda: point_to_grid_module(samples_flat, samples_flat, samples_batch, ngrid))
		i_z_cuda, time_cuda = time_cuda_function(lambda: point_to_grid_module(samples_flat, samples_flat, samples_batch, ngrid, slope=slope))
		print(f"Time for forward pass: {time_cuda}")
		print(f"Batching works correctly: {not torch.allclose(i_z_cuda[:ngrid**3], i_z_cuda[ngrid**3:])}")

		print(f"Average distance and std to point_to_grid features: {(i_z_cuda - i_z_cuda_gt).norm(dim=1).mean()}, {(i_z_cuda - i_z_cuda_gt).norm(dim=1).std()}")

		# Interpolate back
		sample_features = LinearInterpolationFunction.apply(i_z_cuda, samples_flat, samples_batch, ngrid)
		print(f"Average distance and std to original features: {(sample_features - samples_flat).norm(dim=1).mean()}, {(sample_features - samples_flat).norm(dim=1).std()}")

		# Check gradient		
		def fun(*x):
			return PointToGridFunction.apply(*x, samples_batch, ngrid, None, None, None, slope)
		is_jac_correct_cuda, time_gradcheck_cuda = time_cuda_function(lambda: gradcheck(fun, [samples_flat, samples_flat]))

		print(f"Gradient is correct cuda: {is_jac_correct_cuda}")
		pass

	test_linear_interpolation()
	test_point_to_grid()
	test_point_to_grid_masked()
	test_point_to_grid_masked_weighted()
	test_point_to_grid_min_weight()
	test_point_to_grid_sloped()
	pass
