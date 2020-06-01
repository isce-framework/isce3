import numpy as np
from pybind_isce3 import focus

def test_rangecomp():
    nchirp = ndata = 1
    batch = 10
    h = np.ones(nchirp, dtype='c8')

    rc = focus.RangeComp(h, ndata, maxbatch=batch)

    assert rc.chirp_size == nchirp
    assert rc.input_size == ndata
    assert rc.mode == focus.RangeComp.Mode.Full
    assert rc.fft_size >= nchirp + ndata - 1
    assert rc.maxbatch == batch
    assert rc.output_size == nchirp + ndata - 1

    x = np.ones(ndata, dtype='c8')
    y = np.zeros_like(x)
    rc.rangecompress(y, x)
    assert np.allclose(y, x)

    x = np.arange(batch, dtype='c8').reshape((batch, 1))
    y = np.zeros_like(x)
    rc.rangecompress(y, x)
    assert np.allclose(y, x)
