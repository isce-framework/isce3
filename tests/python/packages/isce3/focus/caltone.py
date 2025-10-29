import numpy as np
from isce3.focus import ToneRemover

def test_caltone_removal():
    f = -0.45  # 5% from Nyquist
    n = 1013  # prime for partial blocks
    k = np.arange(n)
    tone = np.exp(1j * 2 * np.pi * f * k)
    # slow sinusoidal amplitude and quadratic phase
    modulation = (1.0 + 0.5 * np.cos(2 * np.pi / n * k)) * np.exp(1j * np.pi * (k / n)**2)
    modulated_tone = tone * modulation

    remover = ToneRemover(f, n)
    estimated = remover.synthesize(remover.analyze(modulated_tone))

    np.testing.assert_array_less(np.abs(np.abs(estimated) - np.abs(modulated_tone)), 0.02)
    np.testing.assert_array_less(np.abs(np.angle(estimated * modulated_tone.conjugate())), 0.1)

    removed = remover.remove_tone(modulated_tone)
    np.testing.assert_array_less(np.abs(removed), 0.1)
