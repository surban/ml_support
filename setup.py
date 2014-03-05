try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='gpuinterop',
      version='0.1',
      description="Interoperability package for PyCUDA, cudamat and Theano",
      author='Sebastian Urban',
      license='BSD',
      py_modules=['gpuinterop'],
      )
