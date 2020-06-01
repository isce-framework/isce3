import numpy as np

complex32 = np.dtype([('r', np.float16), ('i', np.float16)])

def to_complex32(z: np.array):
    zf = np.zeros(z.shape, dtype=complex32)
    zf['r'] = z.real
    zf['i'] = z.imag
    return zf
