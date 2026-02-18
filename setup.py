from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os
import nanobind

setup(
    name='torch_gs',
    packages=['torch_gs'],
    ext_modules=[
        CppExtension(
            'torch_gs._C', 
            ['torch_gs/csrc/rasterizer.cpp'],
            include_dirs=[os.path.join(os.getcwd(), 'torch_gs/csrc')]
        ),
        Extension(
            'torch_gs._MPS',
            sources=[
                'torch_gs/csrc/rasterizer_mps.mm',
                os.path.join(nanobind.source_dir(), 'nb_combined.cpp')
            ],
            include_dirs=[
                nanobind.include_dir(),
                os.path.join(os.path.dirname(nanobind.include_dir()), 'ext', 'robin_map', 'include')
            ],
            extra_compile_args=['-std=c++17', '-O3', '-fPIC', '-fvisibility=hidden', '-x', 'objective-c++', '-fno-objc-arc'],
            extra_link_args=['-framework', 'Metal', '-framework', 'Foundation'],
            language='c++'
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
