from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mymwt',
    ext_modules=[
        CUDAExtension('mymwt', [
            'mymwt.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })