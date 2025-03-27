from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='deep_inverse_hough',
    ext_modules=[
        CUDAExtension(
            name='deep_inverse_hough',
            sources=[
                'deep_inverse_hough_cuda.cpp',
                'deep_inverse_hough_cuda_kernel.cu'
            ],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-arch=sm_75']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
