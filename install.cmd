@echo off

echo Installing cudamat
pushd cudamat
call pip install --user -e .
if not errorlevel 0 goto :eof
popd

echo Installing gnumpy
pushd gnumpy
call pip install --user -e .
if not errorlevel 0 goto :eof
popd

echo Installing PyCUDA
pushd pycuda
call install.cmd
if not errorlevel 0 goto :eof
popd

echo Installing Theano
pushd Theano
call pip install --user -e .
if not errorlevel 0 goto :eof
popd

echo Installing gpuinterop
call pip install --user -e .
if not errorlevel 0 goto :eof
popd


echo ml_support was installed successfully.


