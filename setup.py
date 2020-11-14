from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(name='regev_python',
      version='0.1',
      ext_modules=cythonize('serialize.pyx'),
      include_dirs=[numpy.get_include()])
