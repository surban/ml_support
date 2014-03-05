"""Interoperability package for PyCUDA, cudamat and Theano. Import this module before importing
   Theano, cudamat, gnumpy or PyCUDA to enable interoperability between these packages.
   gpuinterop.using_gpu is True if the GPU is available and used."""

import sys
import logging
import os

# logging
#log_level = logging.DEBUG
log_level = logging.WARNING
logging.basicConfig(level=log_level)
log = logging.getLogger("gpuinterop")

# check if any module has already been imported
gpu_imported = ('theano' in sys.modules or 'cudamat' in sys.modules or
                'gnumpy' in sys.modules or 'pycuda' in sys.modules)
if gpu_imported:
    raise ImportError('gpuinterop must be imported before Theano, cudamat, gnumpy, pycuda')

def set_gnumpy_env():
    if theano.config.device == 'gpu':
        os.environ['GNUMPY_USE_GPU'] = 'yes'
    else:
        os.environ['GNUMPY_USE_GPU'] = 'no'

    if theano.config.floatX == 'float32':
        os.environ['GNUMPY_CPU_PRECISION'] = '32'
    elif theano.config.floatX == 'float64':
        os.environ['GNUMPY_CPU_PRECISION'] = '64'
    else:
        assert False

# the import order is important
import_success = False
try:
    log.debug("importing pycuda.autoinit")
    import pycuda.autoinit
    log.debug("importing pycuda.gpuarray")
    import pycuda.gpuarray
    log.debug("importing cudamat")
    import cudamat
    log.debug("importing theano.sandbox.cuda")
    import theano.sandbox.cuda
    log.debug("importing theano")
    import theano
    set_gnumpy_env()
    log.debug("importing gnumpy")
    import gnumpy
    log.debug("importing theano.tensor")
    import theano.tensor
    log.debug("importing theano.sandbox.cuda")
    import theano.sandbox.cuda
    log.debug("importing theano.misc.gnumpy_utils")
    import theano.misc.gnumpy_utils as gput
 
    import_success = True
except Exception as e:
    import_excpt = e
    log.debug("import failed: " + str(e))

    import theano
    set_gnumpy_env()
    import gnumpy

# test if gpu is available and also used by Theano
using_gpu = (theano.config.device == 'gpu')
if using_gpu and not import_success:
    raise ImportError("Theano is configured to use the GPU but importing GPU modules failed: " + str(e))
log.debug("gpuinterop.using_gpu=%s" % str(using_gpu))
