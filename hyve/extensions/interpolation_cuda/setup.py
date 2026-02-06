from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='linear_interpolation_cuda',
      ext_modules=[cpp_extension.CppExtension('linear_interpolation_cuda', ["linear_interpolation_cuda.cpp", "linear_interpolation_cuda_kernel.cu"])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

setup(name='point_to_grid_cuda',
      ext_modules=[cpp_extension.CppExtension('point_to_grid_cuda', ["point_to_grid_cuda.cpp", "point_to_grid_cuda_kernel.cu"])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})