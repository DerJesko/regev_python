from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(name='regev_python',
      description='A python implementation of the packed Regev encryption scheme',
      author='Jesko Dujmovic',
      version='0.1',
      ext_modules=cythonize('serialize.pyx'),
      include_dirs=[numpy.get_include()])
