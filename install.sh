#!/bin/bash

set -e

echo Installing cudamat
pushd cudamat
pip install --user -e .
popd

echo Installing gnumpy
pushd gnumpy
pip install --user -e .
popd

echo Installing PyCUDA
pushd pycuda
./install.sh
popd

echo Installing Theano
pushd Theano
pip install --user -e .
popd

echo Installing gpuinterop
pip install --user -e .


echo ml_support was installed successfully.


