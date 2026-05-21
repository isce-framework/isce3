import numpy as np

def pow2db(x):
    return 10 * np.log10(x)

def abs2(z):
    return z.real**2 + z.imag**2

def amp2db(z):
    return pow2db(abs2(z))