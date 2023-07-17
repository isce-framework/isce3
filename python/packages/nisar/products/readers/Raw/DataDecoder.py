import logging
from isce3.io import decode_bfpq_lut
from isce3.core.types import complex32, read_c4_dataset_as_c8
import numpy as np

# TODO some CSV logger
log = logging.getLogger("Raw")

class DataDecoder(object):
    """Handle the various data types floating around for raw data, currently
    complex32, complex64, and lookup table.  Indexing operatations always return
    data converted to complex64.
    """
    def __getitem__(self, key):
        return self.decoder(key)

    def _decode_lut(self, key):
        z = self.dataset[key]
        assert self.table is not None
        # Only have 2D version in C++, fall back to Python otherwise
        if z.ndim == 2:
            return decode_bfpq_lut(self.table, z)
        else:
            return self.table[z['r']] + 1j * self.table[z['i']]

    def __init__(self, h5dataset):
        self.table = None
        self.decoder = lambda key: self.dataset[key]
        self.dataset = h5dataset
        self.shape = self.dataset.shape
        self.ndim = self.dataset.ndim
        self.dtype = np.dtype('c8')
        # h5py 3.8.0 returns a compound datatype when accessing a complex32
        # dataset's dtype (https://github.com/h5py/h5py/pull/2157).
        # Previous versions of h5py raise TypeError when attempting to
        # get the dtype. If such exception was raised, we assume the
        # datatype was complex32
        try:
            self.dtype_storage = self.dataset.dtype
        except TypeError:
            self.dtype_storage = complex32
        group = h5dataset.parent
        if "BFPQLUT" in group:
            assert group["BFPQLUT"].dtype == np.float32
            self.table = np.asarray(group["BFPQLUT"])
            self.decoder = self._decode_lut
            log.info("Decoding raw data with lookup table.")
        elif self.dtype_storage == complex32:
            self.decoder = lambda key: read_c4_dataset_as_c8(self.dataset, key)
            log.info("Decoding raw data from float16 encoding.")
        elif self.dtype_storage == np.complex64:
            log.info("Decoding raw data not required")
        else:
            raise ValueError(f"Unsupported raw data type {self.dtype_storage}")
