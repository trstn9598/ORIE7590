# setup.py
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
setup(ext_modules = cythonize(Extension(
    'bd_sim_cython',
    sources=['bd_sim_cython.pyx'],
    language='c++',
    extra_compile_args=['-std=c++17', '-O3', "-Xpreprocessor", '-fopenmp'],
    extra_link_args=['-lomp']
    #include_dirs=[numpy.get_include()],
    #library_dirs=[],
    #libraries=[],
)))