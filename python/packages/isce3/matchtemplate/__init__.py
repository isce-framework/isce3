'''
Amplitude cross-correlation (ampcor) for dense offsets estimation

The implementation is provided by the external `pycuampcor` package; this
module re-exports its CPU implementation for backward compatibility.
'''

try:
    from pycuampcor import PyCPUAmpcor
except ImportError as err:
    _import_error = err

    def __getattr__(name):
        if name == "PyCPUAmpcor":
            raise ImportError("PyCPUAmpcor requires the pycuampcor package "
                              "(conda install -c conda-forge pycuampcor)") \
                from _import_error
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
