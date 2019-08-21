from distutils.core import setup
from Cython.Build import cythonize

setup(name="denemeo",ext_modules = cythonize('cnnlayerforwardnumdiffdeneme.pyx'))

