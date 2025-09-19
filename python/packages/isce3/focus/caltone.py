import numpy as np
from isce3.signal import cola_windows

class ToneRemover:
    def __init__(self, frequency, n, window_size=64, dtype=np.complex64):
        """
        Parameters
        ----------
        frequency : float
            (f_caltone - fc) / fs
        n : int
        window_size : int, optional
        dtype : numpy.dtype
        """
        self.dtype = dtype
        self.size = n
        # NOTE Use a consistent phase reference across blocks.
        tone = np.exp(-1j * 2 * np.pi * frequency * np.arange(n))
        self.wavelets = []
        for block, window in cola_windows(n, window_size):
            analysis = (window * tone[block] / np.sum(window)).astype(dtype)
            synthesis = (window * tone[block]).conjugate().astype(dtype)
            self.wavelets.append((block, analysis, synthesis))

    def analyze(self, z):
        if len(z) != self.size:
            raise ValueError(
                f"Planned for length={self.size} but got {len(z)}")
        return np.array([w.dot(z[block]) for (block, w, _) in self.wavelets])

    def synthesize(self, coeffs):
        if len(coeffs) != len(self.wavelets):
            raise ValueError("Need one coefficient per wavelet.")
        z = np.zeros(self.size, self.dtype)
        for i, (block, _, w) in enumerate(self.wavelets):
            z[block] += coeffs[i] * w
        return z

    def remove_tone(self, z):
        if z.ndim != 1:
            raise NotImplementedError("Only 1D estimation is implemented")
        coeffs = self.analyze(z)
        return z - self.synthesize(coeffs)
