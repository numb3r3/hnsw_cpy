import os
from setuptools import setup, find_packages
from setuptools.extension import Extension

root = os.path.abspath(os.path.dirname(__file__))

setup(
    name='hnsw_cpy',
    packages=find_packages(),
    version='0.0.1',
    zip_safe=False,
    setup_requires=[
        'setuptools>=18.0',
        'cython',
    ],
    ext_modules=[
        Extension(
            'hnsw_cpy.cython_core.hnsw',
            ['hnsw_cpy/cython_core/hnsw.pyx'],
            extra_compile_args=[
                '-O3'],
            language="c++",
        ),
        Extension(
            'hnsw_cpy.cython_core.utils',
            ['hnsw_cpy/cython_core/utils.pyx'],
            extra_compile_args=['-O3'],
        ),
    ],
    install_requires=[
        'termcolor>=1.1',
    ],
    extras_require={
        'test': ['numpy'],
    },
)
