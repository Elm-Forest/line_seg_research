from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='deep_hough_plus',
    ext_modules=[
        CUDAExtension(
            name='deep_hough_plus',
            sources=[
                'deep_hough_plus_cuda.cpp',
                'deep_hough_plus_cuda_kernel.cu'
            ],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-arch=sm_75']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
