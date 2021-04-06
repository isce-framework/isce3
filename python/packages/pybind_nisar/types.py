import h5py
import numpy as np

complex32 = np.dtype([('r', np.float16), ('i', np.float16)])


def to_complex32(z: np.array):
    zf = np.zeros(z.shape, dtype=complex32)
    zf['r'] = z.real
    zf['i'] = z.imag
    return zf


def read_c4_dataset_as_c8(ds: h5py.Dataset, key=np.s_[...]):
    """
    Read a complex float16 HDF5 dataset as a numpy.complex64 array.

    Avoids h5py/numpy dtype bugs and uses numpy float16 -> float32 conversions
    which are about 10x faster than HDF5 ones.
    """
    # This context manager avoids h5py exception:
    # TypeError: data type '<c4' not understood
    with ds.astype(complex32):
        z = ds[key]
    # Define a similar datatype for complex64 to be sure we cast safely.
    complex64 = np.dtype([("r", np.float32), ("i", np.float32)])
    # Cast safely and then view as native complex64 numpy dtype.
    return z.astype(complex64).view(np.complex64)
