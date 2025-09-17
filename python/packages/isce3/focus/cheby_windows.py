from enum import Enum, unique
import numpy as np
from isce3.core import KernelF32, ChebyKernelF32


# helper classes for fitting

class CosineWindowF32(KernelF32):
    def __init__(self, pedestal_height: float = 1.0):
        width = 1.0
        KernelF32.__init__(self, width)
        self.pedestal_height = pedestal_height
        self.a0 = (1.0 + pedestal_height) / 2
        self.a1 = 1.0 - self.a0
        self.freq = 2 * np.pi

    def __call__(self, t: float) -> np.float32:
        if not (-0.5 <= t <= 0.5):
            return np.float32(0)
        return np.float32(self.a0 - self.a1 * np.cos(self.freq * (t + 0.5)))


class KaiserWindowF32(KernelF32):
    def __init__(self, beta: float = 0.0):
        width = 1.0
        KernelF32.__init__(self, width)
        self.beta = beta
        self.scale = 1.0 / np.i0(beta)

    def __call__(self, t: float) -> np.float32:
        if not (-0.5 <= t <= 0.5):
            return np.float32(0)
        x = self.beta * np.sqrt(1 - 4 * t * t)
        return np.float32(np.i0(x) * self.scale)


@unique
class WindowKind(str, Enum):
    KAISER = "kaiser"
    COSINE = "cosine"


def get_window_approximation(kind: WindowKind, shape, n=8) -> ChebyKernelF32:
    """
    Generate a Chebyshev polynomial that approximates an apodization window.

    Parameters
    ----------
    kind : str or WindowKind
        Either "kaiser" or "cosine" window
    shape : float
        Shape parameter of the window.  Pedestal height for raised cosine
        window or beta for Kaiser window.
    n : int
        Number of coefficients in the polynomial.

    Returns
    -------
    window : isce3.core.ChebyKernelF32
        An approximation to the desired apodization window scaled to the
        domain [-0.5, 0.5].
    """
    kind = WindowKind(kind)
    if kind == "kaiser":
        win = KaiserWindowF32(shape)
    elif kind == "cosine":
        win = CosineWindowF32(shape)
    else:
        assert False, "unhandled WindowKind"
    return ChebyKernelF32(win, n)
