import numpy as np
import numpy.testing as npt
import pytest

from isce3.core import LUT2d
from isce3.signal.rfi_tone_rank import (remove_loud_tones, abs2,
    exp_from_quantile)


@pytest.mark.parametrize("m", [513, 521])
def test_tone_rank(m):
    # repeatable tests, https://www.youtube.com/watch?v=a6iW-8xPw3k
    np.random.seed(12345)

    n = 5557  # prime
    normal = lambda: np.random.normal(size=m * n)
    signal = normal() + 1j * normal()
    signal.shape = (m, n)

    f = 97 / 313  # ratio of different primes
    tone = np.exp(1j * 2 * np.pi * f * np.arange(n))
    phases = np.random.uniform(0, 2 * np.pi, size=m)
    interference = tone[None, :] * np.exp(1j * phases[:, None])

    isr = np.sum(abs2(interference)) / np.sum(abs2(signal))
    isr_desired = 10.0
    z = signal + interference * np.sqrt(isr_desired / isr)

    # Fake Doppler and axes. Data are already baseband.
    t = np.linspace(0, 1, m)
    r = np.linspace(1, 2, n)
    doppler = LUT2d(0.0)

    # Don't bother with dithering stuff--say all samples are valid.
    swaths = np.empty((1, m, 2), int)
    swaths[...] = (0, n)

    block_dims = (256, 512)
    fpr = 0.0005
    _, _, freq, _, _, block_hits = remove_loud_tones(
        z, t, r, swaths, doppler, block_dims,
        nominal_false_positive_rate=fpr, bandwidth=1
    )

    # Check basic bookkeeping.
    assert len(freq) == block_dims[1]
    assert len(freq) == block_hits.shape[2]

    # Make sure we mostly suppressed the correct tone.
    i, j = 0, 10
    many_hits = block_hits[i, j, :] > (10 / m)
    suppressed_freqs = freq[many_hits]
    assert len(suppressed_freqs) > 0
    npt.assert_allclose(suppressed_freqs, f, atol=0.01)

    # If the above passes, then the remaining hits are all false positives.
    # Make sure there aren't a lot more than expected.
    num_false_positives = np.sum(block_hits[i, j, ~many_hits])
    assert num_false_positives / np.prod(block_dims) <= fpr


def rng_lifted_exp(n, λ, bw, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    # First figure out which component of the mixture each sample draws from.
    component = rng.random(n) <= bw
    # Generate n samples from each (a little wasteful).
    zero = np.zeros(n)
    exp = rng.exponential(1 / λ, n)
    # Mix samples in correct proportion.
    return np.where(component, exp, zero)

def pow2db(x):
    return 10 * np.log10(x)

@pytest.mark.parametrize("λ", [0.01, 100.0])
def test_exp_from_quantile(λ):
    rng = np.random.default_rng(12345)  # seed for repeatability
    n = 100_000
    bw = 1 / 1.2
    data = rng_lifted_exp(n, λ, bw, rng)

    median = np.median(data)
    λ_est = exp_from_quantile(0.5, median, bw)
    npt.assert_allclose(pow2db(λ), pow2db(λ_est), atol=1)
