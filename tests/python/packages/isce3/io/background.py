import isce3
import numpy as np
import time
from typing import Tuple

#
# read test adapted from Scott Staniewicz gist at
# https://gist.github.com/scottstanie/867d4d8d5bba508bbc8d528d1d8b52b6
#

work_time = 0.2
io_time = 0.5


class EagerLoader(isce3.io.BackgroundReader):
    def __init__(self, dset, **kw):
        self.dset = dset
        super().__init__(**kw)

    def read(self, s: slice) -> Tuple[slice, np.ndarray]:
        time.sleep(io_time)
        print("read", s)
        return s, self.dset[s]


def work(block):
    print("work", block)
    time.sleep(work_time)


def test_loader():
    dset = np.arange(100).reshape((5, 20))
    loader = EagerLoader(dset)
    slices = [np.s_[i,:] for i in range(dset.shape[0])]
    nslices = len(slices)
    t0 = time.time()

    # start loading first block before processing loop
    loader.queue_read(slices[0])
    for i in range(nslices):
        s, block = loader.get_data()
        assert s == slices[i]
        # Start loading next block while we're working on this one.
        if (i + 1) < nslices:
            loader.queue_read(slices[i+1])
        work(block)

    # sleep to check for race
    time.sleep(io_time + work_time)
    loader.notify_finished()

    elapsed = time.time() - t0
    # make sure we ran faster than serial processing
    assert elapsed < (work_time + io_time) * (dset.shape[0] + 1)


#
# writing is simpler
#

class Writer(isce3.io.BackgroundWriter):
    def __init__(self, dset, **kw):
        self.dset = dset  # memory map, raster, etc.
        super().__init__(**kw)

    def write(self, dest_sel: slice, data: np.ndarray):
        time.sleep(io_time)
        self.dset[dest_sel] = data


def test_writer():
    m, n = 5, 20
    dset = np.zeros((m, n), dtype=int)
    writer = Writer(dset)
    t0 = time.time()

    for i in range(m):
        block = np.arange(n) + i * n
        work(block)
        writer.queue_write(np.s_[i,:], block)

    # sleep to check for race
    time.sleep(io_time + work_time)
    writer.notify_finished()

    elapsed = time.time() - t0

    # make sure blocks are stored in correct order
    expected = np.arange(m * n).reshape((m, n))
    assert np.all(dset == expected)

    # make sure we ran faster than serial processing
    assert elapsed < (work_time + io_time) * (m + 1)


if __name__ == '__main__':
    import logging
    level = logging.DEBUG
    sh = logging.StreamHandler()
    sh.setLevel(level)
    l = logging.getLogger("isce3.io.background")
    l.setLevel(level)
    l.addHandler(sh)
    test_loader()
