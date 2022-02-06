from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cu_sparq',
    ext_modules=[
        CUDAExtension('cu_sparq', [
            'gemm_mt.cpp',
            'gemm_mt_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
