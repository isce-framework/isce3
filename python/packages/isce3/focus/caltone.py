import numpy as np
from isce3.signal import cola_windows

class ToneRemover:
    """
    Coherently estimate and remove a non-stationary tone-like signal.
    """
    def __init__(self, frequency, n, window_size=64, dtype=np.complex64):
        """
        Parameters
        ----------
        frequency : float
            The nominal tone frequency, normalized to radians/sample, e.g.,
            (f_caltone - fc) / fs
            Note there is no check for aliasing.
        n : int
            Number of samples in the signal to analyze.
        window_size : int, optional
            Number of samples in each estimation block.  Estimates will be
            spaced by half the window_size.
        dtype : numpy.dtype, optional
            Data type for filter banks.
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
        """Estimate the time-varying response of a signal `z` to a
        tone-like filter bank.  Akin to a single-bin STFT.
        `z` should be one-dimensional with length self.size.
        """
        if len(z) != self.size:
            raise ValueError(
                f"Planned for length={self.size} but got {len(z)}")
        return np.array([w.dot(z[block]) for (block, w, _) in self.wavelets])

    def synthesize(self, coeffs):
        """Generate a continuous time-domain signal given its filter bank
        response coefficients.  Akin to a single-bin inverse STFT.
        """
        if len(coeffs) != len(self.wavelets):
            raise ValueError("Need one coefficient per wavelet.")
        z = np.zeros(self.size, self.dtype)
        for i, (block, _, w) in enumerate(self.wavelets):
            z[block] += coeffs[i] * w
        return z

    def remove_tone(self, z):
        """Estimate parameters of a time-varying tone and return the signal
        `z` with the tone subtracted.
        """
        if z.ndim != 1:
            raise NotImplementedError("Only 1D estimation is implemented")
        coeffs = self.analyze(z)
        return z - self.synthesize(coeffs)
