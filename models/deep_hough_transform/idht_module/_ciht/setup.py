from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='deep_inverse_hough',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            name='deep_inverse_hough',
            sources=[
                'deep_inverse_hough_cuda.cpp',
                'deep_inverse_hough_cuda_kernel.cu'
            ],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': [
                    '-O2',
                    '-gencode=arch=compute_60,code=sm_60',
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                    '-lineinfo',
                    # 不加 fast-math，以确保 float16 稳定性
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
