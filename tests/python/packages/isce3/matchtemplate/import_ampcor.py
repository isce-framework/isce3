import isce3
import pycuampcor


def test_ampcor_import():
    ampcor = pycuampcor.PyCPUAmpcor()
    # backward compatible alias in isce3
    assert isce3.matchtemplate.PyCPUAmpcor is pycuampcor.PyCPUAmpcor
    if pycuampcor.has_cuda and hasattr(isce3, "cuda"):
        assert isce3.cuda.matchtemplate.PyCuAmpcor is pycuampcor.PyCuAmpcor
