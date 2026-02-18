from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import sys

# Custom build_ext to handle .mm files
class CustomBuildExt(build_ext):
    def build_extensions(self):
        # Add .mm to the compiler's list of valid source extensions if not present
        if hasattr(self.compiler, 'src_extensions'):
             if '.mm' not in self.compiler.src_extensions:
                self.compiler.src_extensions.append('.mm')
        
        # Monkey patch the compiler's language_map_extensions if it exists (distutils)
        if hasattr(self.compiler, 'language_map'):
            self.compiler.language_map['.mm'] = 'objc++'
            
        super().build_extensions()

class get_nanobind:
    def __init__(self, type):
        self.type = type
    def __str__(self):
        import nanobind
        if self.type == 'include':
            return nanobind.include_dir()
        if self.type == 'src':
            return os.path.join(nanobind.source_dir(), 'nb_combined.cpp')
        return ''

setup(
    name='mlx_gs_c_api',
    version='0.1',
    description='MLX C API Rasterizer (Raw C++)',
    ext_modules=[
        Extension(
            'mlx_gs.renderer.rasterizer_c_api', # Importable name
            sources=[
                'mlx_gs/csrc/rasterizer.cpp',
                str(get_nanobind('src'))
            ],
            include_dirs=[
                str(get_nanobind('include')),
                os.path.join(os.path.dirname(str(get_nanobind('include'))), 'ext', 'robin_map', 'include')
            ], 
            library_dirs=[],
            libraries=[],
            extra_compile_args=['-std=c++17', '-O3', '-fPIC', '-fvisibility=hidden'],
            language='c++'
        ),
        Extension(
            'mlx_gs.renderer._rasterizer_metal',
            sources=[
                'mlx_gs/csrc/rasterizer_metal.mm',
                str(get_nanobind('src'))
            ],
            include_dirs=[
                str(get_nanobind('include')),
                os.path.join(os.path.dirname(str(get_nanobind('include'))), 'ext', 'robin_map', 'include')
            ],
            extra_compile_args=['-std=c++17', '-O3', '-fPIC', '-fvisibility=hidden', '-fno-objc-arc'],
            extra_link_args=['-framework', 'Metal', '-framework', 'Foundation'],
            language='c++'
        )
    ],
    cmdclass={'build_ext': CustomBuildExt},
    install_requires=['nanobind']
)
