from Cython.Build import cythonize
from setuptools import setup, find_packages
from setuptools.extension import Extension

extensions = [
    Extension(
        'bindex.cython_core', ['bindex/cython_core/cython_core.pyx'],
    ),
]

setup(
    name='bindex',
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    version='0.0.1',
    zip_safe=False
)
