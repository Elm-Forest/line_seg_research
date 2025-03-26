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
            extra_compile_args={'cxx': ['-O3'], 'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_60,code=sm_60',   # P100 必须有
                    '-gencode=arch=compute_75,code=sm_75',   # T4等
                ],}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
