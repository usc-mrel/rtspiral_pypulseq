# Updated setup.py for MacOS stock LLVM compilation, but I would like to
# add some type of homebrew gcc detection for openmp . . .
import os, sys

def is_platform_windows():
    return sys.platform == "win32"

def is_platform_mac():
    return sys.platform == "darwin"

try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Build import cythonize
import numpy


sourcefiles = ['gropt.pyx', './c/cvx_matrix.c', './c/te_finder.c', 
               './c/op_gradient.c', './c/op_maxwell.c', './c/op_bval.c', 
               './c/op_beta.c', './c/op_eddy.c', './c/op_slewrate.c', 
               './c/op_moments.c', './c/op_pns.c']

include_dirs = [".",  "./c", numpy.get_include()]
library_dirs = [".", "./c"]
if is_platform_windows:
    extra_compile_args = []
else:
    extra_compile_args = ['-std=c11']


extensions = [Extension("cgropt",
                sourcefiles,
                language = "c",
                include_dirs = include_dirs,
                library_dirs = library_dirs,
                extra_compile_args = extra_compile_args,
            )]

setup(
    ext_modules = cythonize(extensions)
)