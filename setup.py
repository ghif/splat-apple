from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

setup(
    name='torch_gs',
    packages=['torch_gs'],
    ext_modules=[
        CppExtension(
            'torch_gs._C', 
            ['torch_gs/csrc/rasterizer.cpp'],
            include_dirs=[os.path.join(os.getcwd(), 'torch_gs/csrc')]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
