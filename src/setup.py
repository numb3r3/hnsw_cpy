import os
from setuptools import setup, find_packages
from setuptools.extension import Extension

root = os.path.abspath(os.path.dirname(__file__))

setup(
    name='bindex',
    packages=find_packages(),
    version='0.0.1',
    zip_safe=False,
    setup_requires=[
        'setuptools>=18.0',
        'cython',
    ],
    ext_modules=[
        Extension(
            'bindex.cython_core',
            ['bindex/cython_core/cython_core.pyx'],
            extra_compile_args=['-O3'],
        ),
        Extension(
            'bindex.cython_hnsw.hnsw',
            ['bindex/cython_hnsw/hnsw.pyx'],
            extra_compile_args=['-O3'],
            language="c++",
        ),
        Extension(
            'bindex.cython_hnsw.utils',
            ['bindex/cython_hnsw/utils.pyx'],
            extra_compile_args=['-O3'],
        ),
        Extension(
            'bindex.cython_lib',
            ['bindex/cython_lib/queue.pyx'],
            extra_compile_args=['-O3'],
        ),
    ],
    install_requires=[
        'termcolor>=1.1',
        'cymem>=2.0.0'
    ],
    extras_require={
        'test': ['numpy'],
    },
)
