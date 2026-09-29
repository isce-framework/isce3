'''
Amplitude cross-correlation (ampcor) for dense offsets estimation with CUDA

The implementation is provided by the external `pycuampcor` package; this
module re-exports its CUDA implementation for backward compatibility.
'''

try:
    from pycuampcor import PyCuAmpcor
except ImportError as err:
    _import_error = err

    def __getattr__(name):
        if name == "PyCuAmpcor":
            raise ImportError("PyCuAmpcor requires the pycuampcor package "
                              "built with CUDA support (conda install -c "
                              "conda-forge 'pycuampcor=*=cuda*')") \
                from _import_error
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
