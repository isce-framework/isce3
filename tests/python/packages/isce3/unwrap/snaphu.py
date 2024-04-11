import os
import tempfile
from typing import Optional

import journal
import numpy as np
import pytest
from numpy.typing import DTypeLike

import isce3
from isce3.unwrap import snaphu


# Set journal application name and increase output verbosity level.
journal.application("snaphu.py")
journal.chronicler.detail = 2


def simulate_terrain(
    length: int,
    width: int,
    *,
    scale: float = 1000.0,
    smoothness: float = 0.8,
    seed: Optional[int] = None,
):
    """Simulate topography using the Diamond-Square algorithm [1]_.

    Parameters
    ----------
    length, width : int
        Output dimensions.
    scale : float
        Controls the range of variation in height. Must be positive.
        (default: 1000.0)
    smoothness : float
        Smoothness constant. Must be in the range [0, 1]. Larger values yield
        smoother terrain. Smaller values result in more jagged terrain.
        (default: 0.8)
    seed : int or None, optional
        Seed for initializing pseudo-random number generator state. Must be
        nonnegative. If None, then the generator will be initialized randomly.
        (default: None)

    Returns
    -------
    z : numpy.ndarray
        Height map.

    References
    ----------
    .. [1] Miller, Gavin S. P., "The definition and rendering of terrain maps".
       ACM SIGGRAPH Computer Graphics. 20(4): 39-48.
    """
    # Validate inputs.
    if (length <= 0) or (width <= 0):
        raise ValueError("output array dimensions must be positive")
    if scale <= 0.0:
        raise ValueError("scale factor must be positive")
    if not (0.0 <= smoothness <= 1.0):
        raise ValueError("smoothness constant must be between 0.0 and 1.0")

    # Setup a pseudo-random number generator.
    rng = np.random.default_rng(seed)

    # For positive n, returns the smallest power of two that is >= n
    def next_power_of_two(n):
        return 1 << (n - 1).bit_length()

    # The algorithm operates on a square array with length,width = 2^n + 1
    size = next_power_of_two(max(length, width) - 1) + 1
    z = np.zeros((size, size))

    # Proceed with alternating iterations of "diamond" and "square" steps with
    # progressively smaller strides. The stride is halved at each iteration
    # until, in the last iteration, it's 2 pixels.
    stride = size - 1
    while stride > 1:
        # In the diamond step, the pixel at the midpoint of each square is set
        # to the average value of the square's four corner pixels plus a random
        # offset.

        # Compute the average of the four corner pixels for each square.
        stop = size - stride
        uli, ulj = np.ogrid[0:stop:stride, 0:stop:stride]
        uri, urj = np.ogrid[0:stop:stride, stride:size:stride]
        lli, llj = np.ogrid[stride:size:stride, 0:stop:stride]
        lri, lrj = np.ogrid[stride:size:stride, stride:size:stride]
        avg = 0.25 * (z[uli, ulj] + z[uri, urj] + z[lli, llj] + z[lri, lrj])

        # Set the midpoint pixel to the average of the four corner pixels plus
        # some random value.
        start = stride // 2
        di, dj = np.ogrid[start:size:stride, start:size:stride]
        rval = scale * rng.uniform(-0.5, 0.5, size=(z[di, dj].shape))
        z[di, dj] = avg + rval

        # In the square step, the pixel at the midpoint of each diamond is set
        # to the average value of the diamond's four corner pixels plus a random
        # offset. This step is a bit more complicated since (A) the pixels of
        # interest form sort of a checkerboard pattern rather than a regular
        # grid, and (B) points located on the border have only three neighboring
        # pixels, not four. (A) is resolved by splitting this update into two
        # separate steps, each assigning values to a different set of pixels. We
        # address (B) by letting the indices wrap around so a fourth neighbor is
        # chosen from the opposing side of the array.

        # For the first set of diamonds, compute the average of the four corner
        # pixels.
        ni, nj = uli, np.append(ulj, [[size - 1]], axis=1)
        si, sj = lli, np.append(llj, [[size - 1]], axis=1)
        ei, ej = di, np.append([[-start]], dj, axis=1)
        wi, wj = di, np.append(dj, [[-start]], axis=1)
        avg = 0.25 * (z[ni, nj] + z[si, sj] + z[ei, ej] + z[wi, wj])

        # Set the midpoint pixel to the average of the four corner pixels plus
        # some random value.
        si, sj = np.ogrid[start:size:stride, 0:size:stride]
        rval = scale * rng.uniform(-0.5, 0.5, size=(z[si, sj].shape))
        z[si, sj] = avg + rval

        # For the second set of diamonds, compute the average of the four corner
        # pixels.
        ni, nj = np.append([[-start]], di, axis=0), dj
        si, sj = np.append(di, [[-start]], axis=0), dj
        ei, ej = np.append(uli, [[size - 1]], axis=0), ulj
        wi, wj = np.append(uri, [[size - 1]], axis=0), urj
        avg = 0.25 * (z[ni, nj] + z[si, sj] + z[ei, ej] + z[wi, wj])

        # Set the midpoint pixel to the average of the four corner pixels plus
        # some random value.
        si, sj = np.ogrid[0:size:stride, start:size:stride]
        rval = scale * rng.uniform(-0.5, 0.5, size=(z[si, sj].shape))
        z[si, sj] = avg + rval

        # At each iteration, the magnitude of the random value is reduced and
        # the stride is halved.
        scale *= 0.5 ** smoothness
        stride //= 2

    # Crop the output array to the desired dimensions.
    return z[:length, :width]


def jaccard_similarity(a, b):
    """Compute the Jaccard similarity coefficient (intersect-over-union) of two
    boolean arrays.

    Parameters
    ----------
    a, b : array_like
        Binary masks.

    Returns
    -------
    J : float
        Jaccard similarity coefficient.
    """
    return np.sum(a & b) / np.sum(a | b)


def simulate_phase_noise(corr, nlooks: float, *, seed: Optional[int] = None):
    """Generate pseudo-random noise samples that approximately match the
    expected distribution of multilooked interferogram phase.

    The resulting samples are zero-mean Gaussian distributed, with variance
    equal to the Cramer-Rao bound of the Maximum Likelihood Estimator for the
    interferometric phase [1]_. This simple approximation is most accurate for
    high coherence and large number of looks. The true phase difference
    distribution is more complicated [2]_.

    Parameters
    ----------
    corr : array_like
        Interferometric correlation magnitude.
    nlooks : float
        Number of independent looks.
    seed : int or None, optional
        Seed for initializing pseudo-random number generator state. Must be
        nonnegative. If None, then the generator will be initialized randomly.
        (default: None)

    Returns
    -------
    phi : array_like
        Phase noise samples.

    References
    ----------
    .. [1] E. Rodriguez, and J. M. Martin, "Theory and design of interferometric
       synthetic aperture radars," IEE Proceedings-F, vol. 139, no. 2, pp.
       147-159, April 1992.
    .. [2] J. S. Lee, K. W. Hoppel, S. A. Mango, and A. R. Miller, "Intensity
       and phase statistics of multilook polarimetric and interferometric SAR
       imagery," IEEE Trans. Geosci. Remote Sens. 32, 1017-1028 (1994).
    """
    # Setup a pseudo-random number generator.
    rng = np.random.default_rng(seed)

    # Approximate interferometric phase standard deviation using a simple
    # approximation that holds for high coherence/number of looks.
    sigma = 1.0 / np.sqrt(2.0 * nlooks) * np.sqrt(1.0 - corr ** 2) / corr

    # Generate zero-mean Gaussian-distributed phase noise samples.
    return rng.normal(scale=sigma)


class TestSnaphu:
    @pytest.mark.parametrize("init_method", ["mcf", "mst"])
    def test_smooth_cost(self, init_method):
        """Test SNAPHU unwrapping using "smooth" cost mode."""
        # Interferogram dimensions
        l, w = 1100, 256

        # Simulate 2-D unwrapped phase field, in radians.
        x = np.linspace(0.0, 50.0, w, dtype=np.float32)
        y = np.linspace(0.0, 50.0, l, dtype=np.float32)
        phase = x + y[:, None]

        # Interferogram with a linear diagonal phase gradient
        igram = np.exp(1j * phase)

        # A tall "U"-shaped region
        mask1 = np.zeros((l, w), dtype=bool)
        mask1[100:900, 50:100] = True
        mask1[100:900, 150:200] = True
        mask1[900:950, 50:200] = True

        # A disjoint rectangular-shaped region
        mask2 = np.zeros((l, w), dtype=bool)
        mask2[1000:1050, 50:200] = True

        # Simulate correlation magnitude with some islands of high-coherence
        # surrounded by low-coherence pixels.
        corr = np.zeros((l, w), dtype=np.float32)
        corr[mask1 | mask2] = 1.0

        # Convert the input arrays into rasters.
        igram_raster = isce3.io.gdal.Raster(igram)
        corr_raster = isce3.io.gdal.Raster(corr)

        # Associate each connected component with its expected integer label.
        labeled_regions = {
            1: mask1,
            2: mask2,
        }

        # Create output rasters for unwrapped phase & connected component
        # labels.
        unw_raster = isce3.io.gdal.Raster("unw.tif", w, l, np.float32, "GTiff")
        ccl_raster = isce3.io.gdal.Raster("ccl.tif", w, l, np.uint32, "GTiff")

        # Unwrap phase using SNAPHU "smooth" cost mode. The connected component
        # labelling is sensitive to cost magnitude. For this problem, we need to
        # increase the cost threshold to get the expected connected components.
        conncomp_params = snaphu.ConnCompParams(cost_thresh=1000)
        snaphu.unwrap(
            unw_raster,
            ccl_raster,
            igram_raster,
            corr_raster,
            nlooks=20.0,
            cost="smooth",
            init_method=init_method,
            conncomp_params=conncomp_params,
        )

        # Check the unwrapped phase. It and the true phase should agree up to
        # some fixed offset within each connected component.
        for _, mask in labeled_regions.items():
            mphase = phase[mask]
            munw = unw_raster.data[mask]
            offset = mphase[0] - munw[0]
            assert np.allclose(mphase - offset, munw, rtol=1e-6, atol=1e-6)

        # Check the set of unique labels (masked-out pixels are labeled 0).
        unique_cc_labels = set(np.unique(ccl_raster.data))
        ref_labels = set(labeled_regions.keys())
        assert unique_cc_labels == ref_labels | {0}

        # Check the region associated with each conncomp label. SNAPHU tends to
        # smooth down the sharp corners a bit compared to our input masks, so we
        # allow for some error.
        for label, mask in labeled_regions.items():
            cc = ccl_raster.data == label
            assert jaccard_similarity(cc, mask) > 0.9

    @pytest.mark.parametrize("init_method", ["mcf", "mst"])
    def test_topo_cost(self, init_method):
        """Test SNAPHU unwrapping using "topo" cost mode."""
        # Simulate a topographic interferometric phase signal using notionally
        # NISAR-like 20 MHz L-band parameters.
        bperp = 500.0
        altitude = 750_000.0
        near_range = 900_000.0
        range_spacing = 6.25
        az_spacing = 6.0
        range_res = 7.5
        az_res = 6.6
        wvl = 0.24
        transmit_mode = "repeat_pass"
        inc_angle = np.deg2rad(37.0)

        # Multilooking params.
        nlooks_range = 5
        nlooks_az = 5
        dr = nlooks_range * range_spacing
        da = nlooks_az * az_spacing
        nlooks = dr * da / (range_res * az_res)

        # Multilooked interferogram dimensions.
        l, w = 1024, 512

        # Simulate topographic height map.
        z = simulate_terrain(l, w, scale=5000.0, seed=1234)

        # Simulate expected interferometric phase from topography.
        r = near_range + dr * np.arange(l)
        phase = -4.0 * np.pi / wvl * bperp / r[:, None] * np.sin(inc_angle) * z

        # Correlation coefficient
        corr = np.full((l, w), fill_value=0.7, dtype=np.float32)

        # Add phase noise.
        phase += simulate_phase_noise(corr, nlooks, seed=1234)

        # Create unit-magnitude interferogram.
        igram = np.exp(1j * phase).astype(np.complex64)

        # Convert the input arrays into rasters.
        igram_raster = isce3.io.gdal.Raster(igram)
        corr_raster = isce3.io.gdal.Raster(corr)

        # Create output rasters for unwrapped phase & connected component
        # labels.
        unw_raster = isce3.io.gdal.Raster("unw.tif", w, l, np.float32, "GTiff")
        ccl_raster = isce3.io.gdal.Raster("ccl.tif", w, l, np.uint32, "GTiff")

        # Cost mode configuration parameters.
        cost_params = snaphu.TopoCostParams(
            bperp=bperp,
            near_range=near_range,
            dr=dr,
            da=da,
            range_res=range_res,
            az_res=az_res,
            wavelength=wvl,
            transmit_mode=transmit_mode,
            altitude=altitude,
        )

        # Unwrap phase using SNAPHU "topo" cost mode.
        snaphu.unwrap(
            unw_raster,
            ccl_raster,
            igram_raster,
            corr_raster,
            nlooks=nlooks,
            cost="topo",
            cost_params=cost_params,
            init_method=init_method,
        )

        # Check the connected component labels. There should be a single
        # connected component (with label 1) which contains most pixels. Any
        # remaining pixels should be masked out (with label 0).
        unique_cc_labels = set(np.unique(ccl_raster.data))
        assert (unique_cc_labels == {1}) or (unique_cc_labels == {0, 1})

        # Computes the fraction of nonzero pixels in the input array
        def frac_nonzero(a):
            return np.count_nonzero(a) / np.size(a)

        assert frac_nonzero(ccl_raster.data == 0) < 1e-3

        # Check the unwrapped phase. The test metric is the fraction of
        # correctly unwrapped pixels, i.e. pixels where the unwrapped phase and
        # the true phase agree up to some constant relative phase, excluding
        # masked pixels.
        mask = ccl_raster.data != 0
        mphase = phase[mask]
        munw = unw_raster.data[mask]
        offset = mphase[0] - munw[0]
        good_pixels = np.isclose(mphase - offset, munw, rtol=1e-6, atol=1e-6)
        assert frac_nonzero(~good_pixels) < 1e-3

    def test_single_row_col_mask(self):
        """Test SNAPHU with connected components separated by a single row or
        column of masked pixels.
        """
        # Interferogram dimensions
        l, w = 513, 513

        # Simulate 2-D unwrapped phase field, in radians.
        x = np.linspace(0.0, 50.0, w, dtype=np.float32)
        y = np.linspace(0.0, 50.0, l, dtype=np.float32)
        phase = x + y[:, None]

        # Mask out the middle row & column, as well as each border row & column.
        mask = np.ones((l, w), dtype=np.uint8)
        mask[[0, l // 2, l - 1]] = 0
        mask[:, [0, w // 2, w - 1]] = 0

        # Simulate correlation coefficient.
        corr = np.full((l, w), fill_value=0.7, dtype=np.float32)

        # Add phase noise.
        nlooks = 9.0
        phase += simulate_phase_noise(corr, nlooks, seed=12345)

        # Interferogram with a linear diagonal phase gradient.
        igram = np.exp(1j * phase)

        # Set masked interferogram pixels' magnitude to zero.
        igram[mask == 0] = 0.0

        # Convert the input arrays into rasters.
        igram_raster = isce3.io.gdal.Raster(igram)
        corr_raster = isce3.io.gdal.Raster(corr)
        mask_raster = isce3.io.gdal.Raster(mask)

        # Create output rasters for unwrapped phase & connected component
        # labels.
        unw_raster = isce3.io.gdal.Raster("unw.tif", w, l, np.float32, "GTiff")
        ccl_raster = isce3.io.gdal.Raster("ccl.tif", w, l, np.uint32, "GTiff")

        # Unwrap phase using SNAPHU "smooth" cost mode.
        snaphu.unwrap(
            unw_raster,
            ccl_raster,
            igram_raster,
            corr_raster,
            nlooks=nlooks,
            cost="smooth",
            mask=mask_raster,
        )

        # Check the connected component labels. There should be four distinct regions
        # with unique nonzero labels. Masked pixels should be labeled 0.
        unique_cc_labels = set(np.unique(ccl_raster.data))
        assert unique_cc_labels == {0, 1, 2, 3, 4}
        assert np.all(ccl_raster.data[mask == 0] == 0)

        # Check the unwrapped phase for each connected component. The unwrapped and true
        # phase should agree up to some fixed offset within each region.
        for label in range(1, 5):
            lmask = (ccl_raster.data == label)
            mphase = phase[lmask]
            munw = unw_raster.data[lmask]
            offset = mphase[0] - munw[0]
            assert np.allclose(mphase - offset, munw, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("conncomp_dtype", [np.uint16, np.int32])
    def test_conncomp_dtype(self, conncomp_dtype: DTypeLike):
        # Simulate interferogram containing a diagonal phase ramp with multiple fringes.
        y, x = np.ogrid[-3:3:512j, -3:3:512j]
        phase = np.pi * (x + y)
        igram = np.exp(1j * phase)

        # Simulate sample coherence for an interferogram with no noise.
        corr = np.ones(igram.shape)

        # Create a binary mask that subdivides the array into 4 disjoint quadrants of
        # valid samples separated by a single row & column of invalid samples.
        mask = np.ones(igram.shape, dtype=np.uint8)
        mask[256, :] = 0
        mask[:, 256] = 0

        # Convert the input arrays into rasters.
        igram_raster = isce3.io.gdal.Raster(igram)
        corr_raster = isce3.io.gdal.Raster(corr)
        mask_raster = isce3.io.gdal.Raster(mask)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create output rasters.
            length, width = igram.shape
            unw_raster = isce3.io.gdal.Raster(
                path=os.path.join(tmpdir, "unw.tif"),
                width=width,
                length=length,
                datatype=np.float32,
                driver="GTiff",
            )
            conncomp_raster = isce3.io.gdal.Raster(
                path=os.path.join(tmpdir, "conncomp.tif"),
                width=width,
                length=length,
                datatype=conncomp_dtype,
                driver="GTiff",
            )

            # Unwrap.
            snaphu.unwrap(
                unw=unw_raster,
                conncomp=conncomp_raster,
                igram=igram_raster,
                corr=corr_raster,
                nlooks=1.0,
                mask=mask_raster,
            )

            # Check that the output contains 4 distinct nonzero connected
            # component labels.
            assert set(np.unique(conncomp_raster)) == {0, 1, 2, 3, 4}

    def test_tile_mode(self):
        """Test SNAPHU tiled unwrapping mode."""
        # Interferogram dimensions
        l, w = 1100, 256

        # Simulate 2-D unwrapped phase field, in radians.
        x = np.linspace(0.0, 50.0, w, dtype=np.float32)
        y = np.linspace(0.0, 50.0, l, dtype=np.float32)
        phase = x + y[:, None]

        # Interferogram with a linear diagonal phase gradient
        igram = np.exp(1j * phase)

        # A tall "U"-shaped region
        mask1 = np.zeros((l, w), dtype=bool)
        mask1[100:900, 50:100] = True
        mask1[100:900, 150:200] = True
        mask1[900:950, 50:200] = True

        # A disjoint rectangular-shaped region
        mask2 = np.zeros((l, w), dtype=bool)
        mask2[1000:1050, 50:200] = True

        # Simulate correlation magnitude with some islands of high-coherence
        # surrounded by low-coherence pixels.
        corr = np.full((l, w), fill_value=0.0, dtype=np.float32)
        corr[mask1 | mask2] = 1.0

        # Convert the input arrays into rasters.
        igram_raster = isce3.io.gdal.Raster(igram)
        corr_raster = isce3.io.gdal.Raster(corr)

        # Associate each connected component with its expected integer label.
        labeled_regions = {
            1: mask1,
            2: mask2,
        }

        # Create output rasters for unwrapped phase & connected component
        # labels.
        unw_raster = isce3.io.gdal.Raster("unw.tif", w, l, np.float32, "GTiff")
        ccl_raster = isce3.io.gdal.Raster("ccl.tif", w, l, np.uint32, "GTiff")

        # Unwrap phase using SNAPHU "defo" cost mode.
        tiling_params = snaphu.TilingParams(
            nproc=4, tile_nrows=2, tile_ncols=2, row_overlap=16, col_overlap=16,
        )
        snaphu.unwrap(
            unw_raster,
            ccl_raster,
            igram_raster,
            corr_raster,
            nlooks=20.0,
            cost="defo",
            tiling_params=tiling_params,
        )

        # Check the unwrapped phase. It and the true phase should agree up to
        # some fixed offset within each connected component.
        for _, mask in labeled_regions.items():
            mphase = phase[mask]
            munw = unw_raster.data[mask]
            offset = mphase[0] - munw[0]
            assert np.allclose(mphase - offset, munw, rtol=1e-6, atol=1e-6)
