from setuptools import setup

setup(package_dir={"": "src"},
      cffi_modules=["src/spiralgen_build.py:ffibuilder"])