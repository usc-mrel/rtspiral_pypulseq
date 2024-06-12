 #!/usr/bin/env python

import os
import sys

from setuptools import setup, find_packages, Extension

from Cython.Build import cythonize
import numpy


os.chdir(os.path.dirname(sys.argv[0]) or ".")

def is_platform_windows():
    return sys.platform == "win32"

def is_platform_mac():
    return sys.platform == "darwin"

sourcefiles = ['gropt/gropt.pyx', 'gropt/src/cvx_matrix.c', 'gropt/src/te_finder.c', 'gropt/src/op_gradient.c', 'gropt/src/op_maxwell.c', 'gropt/src/op_bval.c', 'gropt/src/op_beta.c', 'gropt/src/op_eddy.c', 'gropt/src/op_slewrate.c', 'gropt/src/op_moments.c', 'gropt/src/op_pns.c']

include_dirs = ["gropt/",  "gropt/src", numpy.get_include()]
library_dirs = ["gropt/", "gropt/src"]
if is_platform_windows:
    extra_compile_args = []
else:
    extra_compile_args = ['-std=c11']


extensions = [Extension("gropt.gropt",
                sourcefiles,
                language = "c",
                include_dirs = include_dirs,
                library_dirs = library_dirs,
                extra_compile_args = extra_compile_args,
            )]

setup(
    name="rtspiral",
    version="0.1",
    description="An example project using Python's CFFI",
    long_description=open("README.md", "rt").read(),
    url="https://github.com/btasdelen/rtspiral_pypulseq",
    author="Bilal Tasdelen",
    author_email="billtasdelen@gmail.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: PyPy",
        "License :: OSI Approved :: GPL3 License",
    ],
    packages=find_packages(),
    install_requires=["cffi>=1.0.0", "cython"],
    setup_requires=["cffi>=1.0.0", "cython"],
    cffi_modules=[
        "./libvds/vds_build.py:ffibuilder",
    ],
    ext_modules = cythonize(extensions),
)