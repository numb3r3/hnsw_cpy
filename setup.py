from Cython.Build import cythonize
from setuptools import setup, find_packages
from setuptools.extension import Extension

extensions = [
    Extension(
        'fast_index',
        ["fast_bt_index/fast_index.pyx"],
    ),
]

setup(
    name='fast_index',
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    version='0.0.1',
    zip_safe=False
)
