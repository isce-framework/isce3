import h5py
import journal
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

    Parameters
    ----------
    ds: h5py.Dataset
        Path to the c4 HDF5 dataset
    key: numpy.s_
        Numpy slice to subset input dataset

    Returns
    -------
    np.ndarray(numpy.complex64)
        Complex array (numpy.complex64) from sliced input HDF5 dataset
    """
    # This avoids h5py exception:
    # TypeError: data type '<c4' not understood
    # Also note this syntax changed in h5py 3.0 and was deprecated in 3.6, see
    # https://docs.h5py.org/en/stable/whatsnew/3.6.html
    z = ds.astype(complex32)[key]
    # Define a similar datatype for complex64 to be sure we cast safely.
    complex64 = np.dtype([("r", np.float32), ("i", np.float32)])
    # Cast safely and then view as native complex64 numpy dtype.
    return z.astype(complex64).view(np.complex64)


def read_complex_dataset(ds: h5py.Dataset, key=np.s_[...]):
    """
    Read a complex dataset including complex float16 (c4) datasets, in which
    case, the function `read_c4_dataset_as_c8` is called to avoid
    h5py/numpy dtype (TypeError) bugs

    Parameters
    ----------
    ds: h5py.Dataset
        Path to the complex HDF5 dataset
    key: numpy.s_
        Numpy slice to subset input dataset

    Returns
    -------
    np.ndarray
        Complex array from sliced input HDF5 dataset
    """
    try:
        return ds[key]
    except TypeError:
        pass
    return read_c4_dataset_as_c8(ds, key=key)


class ComplexFloat16Decoder:
    """Provide a thin wrapper around an h5py.Dataset that takes care
    to decode complex Float16 data quickly and safely.
    """
    def __init__(self, ds: h5py.Dataset):
        self.dataset = ds
        self.shape = ds.shape
        self.ndim = ds.ndim
        self.chunks = ds.chunks
        self.dtype = np.dtype('c8')
        # No way to safely check, see https://github.com/h5py/h5py/issues/2156
        self.dtype_storage = complex32

    def __getitem__(self, key):
        return read_c4_dataset_as_c8(self.dataset, key)


def truncate_mantissa(z: np.ndarray, significant_bits=10):
    '''
    Zero out bits in mantissa of elements of array in place.

    Parameters
    ----------
    z: numpy.array
        Real or complex array whose mantissas are to be zeroed out
    nonzero_mantissa_bits: int, optional
        Number of bits to preserve in mantisa. Defaults to 10.
    '''
    error_channel = journal.info("truncate_mantissa")

    # check if z is np.ndarray
    if not isinstance(z, np.ndarray):
        err_str = 'argument z is not of type np.ndarray'
        error_channel.log(err_str)
        raise ValueError(err_str)

    # recurse for complex data
    if np.iscomplexobj(z):
        truncate_mantissa(z.real, significant_bits)
        truncate_mantissa(z.imag, significant_bits)
        return

    if not issubclass(z.dtype.type, np.floating):
        err_str = 'argument z is not complex float or float type'
        error_channel.log(err_str)
        raise ValueError(err_str)

    mant_bits = np.finfo(z.dtype).nmant
    float_bytes = z.dtype.itemsize

    if significant_bits == mant_bits:
        return

    if not 0 < significant_bits <= mant_bits:
        err_str = f'Require 0 < {significant_bits=} <= {mant_bits}'
        error_channel.log(err_str)
        raise ValueError(err_str)

    # create integer value whose binary representation is one for all bits in
    # the floating point type.
    allbits = (1 << (float_bytes * 8)) - 1

    # Construct bit mask by left shifting by nzero_bits and then truncate.
    # This works because IEEE 754 specifies that bit order is sign, then
    # exponent, then mantissa.  So we zero out the least significant mantissa
    # bits when we AND with this mask.
    nzero_bits = mant_bits - significant_bits
    bitmask = (allbits << nzero_bits) & allbits

    utype = np.dtype(f"u{float_bytes}")
    # view as uint type (can not mask against float)
    u = z.view(utype)
    # bitwise-and in-place to mask
    u &= bitmask
