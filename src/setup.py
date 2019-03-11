import os
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize


root = os.path.abspath(os.path.dirname(__file__))

extensions = [
        Extension(
            'hnsw_cpy.cython_core.hnsw',
            ['hnsw_cpy/cython_core/hnsw.pyx'],
            extra_compile_args=[
                '-O3'],
            language="c++",
        ),
        Extension(
            'hnsw_cpy.cython_core.heappq',
            ['hnsw_cpy/cython_core/heappq.pyx'],
            extra_compile_args=[
                '-O3'],
        ),
        Extension(
            'hnsw_cpy.cython_core.queue',
            ['hnsw_cpy/cython_core/queue.pyx'],
            extra_compile_args=[
                '-O3'],
        ),
]


setup(
    name='hnsw_cpy',
    packages=find_packages(),
    version='0.0.1',
    zip_safe=False,
    setup_requires=[
        'setuptools>=18.0',
        'cython',
    ],
    ext_modules=cythonize(extensions),
    install_requires=[
        'termcolor>=1.1',
    ],
    extras_require={
        'test': ['numpy'],
    },
)
