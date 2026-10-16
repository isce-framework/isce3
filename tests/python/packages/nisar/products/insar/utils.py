"""Regression tests for the mask generation in nisar.products.insar.utils.

Any implementation of generate_insar_mask() (the current scalar
per-pixel loop or a vectorized replacement) must reproduce the same
mask semantics exactly:

- SubSwaths.get_sample_sub_swath equivalence: out-of-bounds -> 0,
  first-match-wins sub-swath ordering, an empty valid-samples array
  claims every in-bounds sample, and a dataset without sub-swath
  information assigns 1 everywhere in bounds;
- the two distinct rounding rules: the sub-swath lookup uses
  int(x + 0.5) (truncation toward zero) while the exception-mask
  lookup uses round() (round-half-even);
- rows/columns outside the reference swath produce whole-pixel 0, and
  secondary indices outside the secondary swath drop the secondary
  exception bits while keeping the sub-swath contribution;
- the inputDataExceptionMask bytes are packed into the uint32 mask as
  (ref << 16) | (sec << 8), which must hold for mask bytes >= 2**(8-k)
  under NumPy 2 scalar promotion (NEP 50) as well.
"""
import numpy as np
import h5py
import pytest
from osgeo import gdal

import isce3
from nisar.products.insar.utils import (_compute_subswath_mask_id,
                                        generate_insar_mask)

# Vectorized implementations carry a _subswath_numbers helper; the
# scalar loop does not. The helper-level tests are skipped when it is
# absent and activate automatically once a vectorization lands.
try:
    from nisar.products.insar.utils import _subswath_numbers
except ImportError:
    _subswath_numbers = None


def unpack_mask(result):
    """Return the uint32 mask from a generate_insar_mask() result.

    Tolerates both the historical single-array return and the
    two-element (mask, pol_valid_mask) tuple return proposed in
    isce-framework/isce3#379, so the regression tests keep guarding the
    mask semantics across either signature; any other arity fails.

    The per-pol validity word is sanity-checked against the mask's
    sub-swath digits before it is set aside: with the uint8 or absent
    exception masks these tests feed, reference (secondary) validity
    bits must be set exactly where the mask's reference (secondary)
    sub-swath digit is nonzero. Which per-pol bits carry that validity
    is #379's own semantics and is left to that PR's tests.
    """
    if not isinstance(result, tuple):
        return result
    mask, pol_valid_mask = result
    assert pol_valid_mask.shape == mask.shape
    assert pol_valid_mask.dtype == np.uint16
    digits = mask & np.uint32(0xFF)
    np.testing.assert_array_equal(
        (pol_valid_mask & np.uint16(0xFF00)) != 0, digits // 10 > 0)
    np.testing.assert_array_equal(
        (pol_valid_mask & np.uint16(0x00FF)) != 0, digits % 10 > 0)
    return mask

SWATH_PATH = "/science/LSAR/RSLC/swaths"

# Unique names for the in-memory h5py files
_h5_counter = 0


class FakeSwath:
    """Duck-typed stand-in for the Swath metadata generate_insar_mask
    reads (lines, samples, sub_swaths)."""

    def __init__(self, lines, samples, subswaths):
        self.lines = lines
        self.samples = samples
        self._subswaths = subswaths

    def sub_swaths(self):
        return self._subswaths


class FakeSLC:
    SwathPath = SWATH_PATH

    def __init__(self, swath):
        self._swath = swath

    def getSwathMetadata(self, freq):
        return self._swath


def make_h5(freq, exception_mask):
    """In-memory h5py file; exception_mask=None omits the dataset."""
    global _h5_counter
    _h5_counter += 1
    f = h5py.File(f"insar-mask-test-{_h5_counter}", "w",
                  driver="core", backing_store=False)
    if exception_mask is not None:
        f.create_dataset(
            f"{SWATH_PATH}/frequency{freq}/inputDataExceptionMask",
            data=exception_mask)
    return f


def make_offset_raster(path, data):
    drv = gdal.GetDriverByName("ENVI")
    ds = drv.Create(str(path), data.shape[1], data.shape[0], 1,
                    gdal.GDT_Float64)
    ds.GetRasterBand(1).WriteArray(data)
    ds.FlushCache()
    ds = None


def make_subswaths(rng, lines, samples, n_sub, empty_at=(), no_info=False):
    """Random sub-swath layout with a few fully-invalid lines; empty_at
    sub-swath numbers get an empty valid-samples array, no_info builds a
    SubSwaths without any sub-swath information."""
    if no_info:
        return isce3.product.SubSwaths(lines, samples, [])
    arrays = []
    for s in range(1, n_sub + 1):
        if s in empty_at:
            arrays.append(np.empty((0, 0), dtype=np.int32))
            continue
        start = rng.integers(0, samples, size=lines)
        width = rng.integers(0, samples // 2 + 1, size=lines)
        end = np.minimum(start + width, samples)
        invalid = rng.random(lines) < 0.1
        end[invalid] = start[invalid]
        arrays.append(np.stack([start, end], axis=1).astype(np.int32))
    return isce3.product.SubSwaths(lines, samples, arrays)


def build_offsets(rng, lines, samples, scale):
    """Offset field with adversarial values: smooth random, offsets that
    land (index + offset) exactly on k + 0.5 half-integers where the two
    rounding rules diverge, and large pushes outside the secondary
    swath."""
    off = rng.normal(0.0, scale, size=(lines, samples))
    jj = np.arange(samples, dtype=np.float64)
    half_rows = rng.choice(lines, size=max(1, lines // 5), replace=False)
    for r in half_rows:
        targets = rng.integers(-3, samples + 3,
                               size=samples).astype(np.float64) + 0.5
        sel = rng.random(samples) < 0.3
        off[r, sel] = (targets - jj)[sel]
    blow = rng.random((lines, samples)) < 0.02
    off[blow] = rng.choice([-1.0, 1.0], size=blow.sum()) * (samples + lines)
    return off


def scalar_reference_mask(ref_swath, sec_swath, ref_exception_mask,
                          sec_exception_mask, range_off, azimuth_off,
                          azi_idx_arr, rg_idx_arr):
    """Per-pixel reference implementation of the intended mask
    semantics: _compute_subswath_mask_id for the sub-swath digits and
    Python-int bit packing of the exception-mask bytes (immune to fixed
    width scalar promotion)."""
    ref_subswaths = ref_swath.sub_swaths()
    sec_subswaths = sec_swath.sub_swaths()
    mask = np.zeros((len(azi_idx_arr), len(rg_idx_arr)), dtype=np.uint32)
    for row, i in enumerate(azi_idx_arr):
        if not (0 <= i < ref_swath.lines):
            continue
        for col, j in enumerate(rg_idx_arr):
            if not (0 <= j < ref_swath.samples):
                continue
            az_off = azimuth_off[int(i), int(j)]
            rg_off = range_off[int(i), int(j)]
            mask_id = _compute_subswath_mask_id(
                int(i), int(j), az_off, rg_off,
                ref_subswaths, sec_subswaths)
            mask_id |= int(ref_exception_mask[int(i), int(j)]) << 16
            sec_i = round(i + az_off)
            sec_j = round(j + rg_off)
            if (0 <= sec_i < sec_swath.lines and
                    0 <= sec_j < sec_swath.samples):
                mask_id |= int(sec_exception_mask[sec_i, sec_j]) << 8
            mask[row, col] = mask_id
    return mask


@pytest.mark.skipif(_subswath_numbers is None,
                    reason="no vectorized _subswath_numbers helper in "
                           "this implementation (if a vectorization is "
                           "merged and this still skips, suspect a helper "
                           "rename or removal)")
class TestSubswathNumbers:
    """_subswath_numbers against the scalar pybind oracle
    SubSwaths.get_sample_sub_swath."""

    @pytest.mark.parametrize("n_sub,empty_at,no_info", [
        (3, (), False),
        (1, (), False),
        (3, (2,), False),
        (0, (), True),
    ])
    def test_matches_scalar_api(self, n_sub, empty_at, no_info):
        rng = np.random.default_rng(12345 + n_sub + 100 * no_info)
        lines, samples = 40, 56
        subswaths = make_subswaths(rng, lines, samples, n_sub,
                                   empty_at=empty_at, no_info=no_info)
        intervals = [subswaths.get_valid_samples_array(s)
                     for s in range(1, subswaths.num_sub_swaths + 1)]

        # every index pair from 4 outside the swath on either side,
        # plus random scattered pairs
        azi = np.arange(-4, lines + 4, dtype=np.int64)
        rg = np.arange(-4, samples + 4, dtype=np.int64)
        azi_grid, rg_grid = np.meshgrid(azi, rg, indexing="ij")

        actual = _subswath_numbers(subswaths, intervals,
                                   azi_grid, rg_grid)

        expected = np.array(
            [[subswaths.get_sample_sub_swath(int(a), int(r)) for r in rg]
             for a in azi], dtype=np.int64)
        np.testing.assert_array_equal(actual, expected)


class TestGenerateInsarMask:
    """generate_insar_mask against the per-pixel scalar reference."""

    @pytest.mark.parametrize(
        "name,seed,ref_dims,sec_dims,n_sub,empty_at,no_info,no_masks,"
        "off_scale",
        [
            ("random_3sub", 1, (40, 56), (37, 59), 3, (), False, False, 2.5),
            ("empty_mid_subswath", 2, (32, 48), (32, 48), 3, (2,), False,
             False, 2.5),
            ("no_subswath_info", 3, (32, 48), (30, 44), 0, (), True, False,
             2.5),
            ("no_exception_masks", 4, (32, 48), (32, 48), 2, (), False, True,
             2.5),
            ("large_offsets", 5, (36, 42), (22, 30), 2, (), False, False,
             25.0),
        ])
    def test_matches_scalar_reference(self, tmp_path, name, seed, ref_dims,
                                      sec_dims, n_sub, empty_at, no_info,
                                      no_masks, off_scale):
        rng = np.random.default_rng(seed)
        ref_lines, ref_samples = ref_dims
        sec_lines, sec_samples = sec_dims

        ref_swath = FakeSwath(
            ref_lines, ref_samples,
            make_subswaths(rng, ref_lines, ref_samples, n_sub,
                           empty_at=empty_at, no_info=no_info))
        sec_swath = FakeSwath(
            sec_lines, sec_samples,
            make_subswaths(rng, sec_lines, sec_samples, n_sub,
                           no_info=no_info))

        if no_masks:
            ref_exc = np.zeros((ref_lines, ref_samples), dtype=np.uint8)
            sec_exc = np.zeros((sec_lines, sec_samples), dtype=np.uint8)
            ref_h5 = make_h5("A", None)
            sec_h5 = make_h5("A", None)
        else:
            ref_exc = rng.integers(0, 256, size=(ref_lines, ref_samples),
                                   dtype=np.uint8)
            sec_exc = rng.integers(0, 256, size=(sec_lines, sec_samples),
                                   dtype=np.uint8)
            ref_h5 = make_h5("A", ref_exc)
            sec_h5 = make_h5("A", sec_exc)

        range_off = build_offsets(rng, ref_lines, ref_samples, off_scale)
        azimuth_off = build_offsets(rng, ref_lines, ref_samples, off_scale)
        rg_off_path = tmp_path / f"{name}_range.off"
        az_off_path = tmp_path / f"{name}_azimuth.off"
        make_offset_raster(rg_off_path, range_off)
        make_offset_raster(az_off_path, azimuth_off)

        # integral-float index arrays extending past the swath on both
        # sides, matching the np.round(...) arrays the callers build
        azi_idx = np.round(np.linspace(-3, ref_lines + 3, ref_lines + 8))
        rg_idx = np.round(np.linspace(-3, ref_samples + 3, ref_samples + 8))

        actual = unpack_mask(generate_insar_mask(
            FakeSLC(ref_swath), FakeSLC(sec_swath), ref_h5, sec_h5,
            str(rg_off_path), str(az_off_path), "A", azi_idx, rg_idx))
        expected = scalar_reference_mask(
            ref_swath, sec_swath, ref_exc, sec_exc, range_off,
            azimuth_off, azi_idx, rg_idx)

        ref_h5.close()
        sec_h5.close()
        assert actual.dtype == np.uint32
        np.testing.assert_array_equal(actual, expected)

    def test_exception_mask_bit_packing(self, tmp_path):
        """Exact uint32 packing, including bytes with the MSB set whose
        << 16 / << 8 shifts overflow a fixed-width uint8 scalar
        (regression test for silently dropped exception bits)."""
        lines, samples = 4, 6
        # single sub-swath covering every sample -> sub-swath digits 11
        full = np.tile(np.array([[0, samples]], dtype=np.int32),
                       (lines, 1))
        swath = FakeSwath(
            lines, samples,
            isce3.product.SubSwaths(lines, samples, [full]))

        ref_exc = np.zeros((lines, samples), dtype=np.uint8)
        sec_exc = np.zeros((lines, samples), dtype=np.uint8)
        ref_exc[1, 2] = 0xAB
        sec_exc[1, 2] = 0xCD
        ref_exc[2, 3] = 0x80
        sec_exc[2, 3] = 0xFF
        ref_h5 = make_h5("A", ref_exc)
        sec_h5 = make_h5("A", sec_exc)

        zeros = np.zeros((lines, samples), dtype=np.float64)
        rg_off_path = tmp_path / "packing_range.off"
        az_off_path = tmp_path / "packing_azimuth.off"
        make_offset_raster(rg_off_path, zeros)
        make_offset_raster(az_off_path, zeros)

        idx_azi = np.arange(lines, dtype=np.float64)
        idx_rg = np.arange(samples, dtype=np.float64)
        mask = unpack_mask(generate_insar_mask(
            FakeSLC(swath), FakeSLC(swath), ref_h5, sec_h5,
            str(rg_off_path), str(az_off_path), "A", idx_azi, idx_rg))
        ref_h5.close()
        sec_h5.close()

        assert mask[1, 2] == (0xAB << 16) | (0xCD << 8) | 11
        assert mask[2, 3] == (0x80 << 16) | (0xFF << 8) | 11
        assert mask[0, 0] == 11

    def test_rounding_rules_diverge(self, tmp_path):
        """The sub-swath lookup truncates int(x + 0.5) while the
        exception-mask lookup rounds half to even; an exact +0.5 range
        offset at an even column exercises both: the sub-swath lookup
        reads column j + 1 while the exception mask reads column j."""
        lines, samples = 3, 8
        # sub-swath 1 = columns [0, 4), sub-swath 2 = columns [4, 8)
        s1 = np.tile(np.array([[0, 4]], dtype=np.int32), (lines, 1))
        s2 = np.tile(np.array([[4, 8]], dtype=np.int32), (lines, 1))
        swath = FakeSwath(
            lines, samples,
            isce3.product.SubSwaths(lines, samples, [s1, s2]))

        sec_exc = np.zeros((lines, samples), dtype=np.uint8)
        sec_exc[1, 2] = 0x11    # read by round-half-even (2.5 -> 2)
        sec_exc[1, 3] = 0x22    # NOT read: int(2.5 + 0.5) = 3 is the
        #                         sub-swath lookup only
        ref_h5 = make_h5("A", np.zeros((lines, samples), dtype=np.uint8))
        sec_h5 = make_h5("A", sec_exc)

        rg_off = np.zeros((lines, samples), dtype=np.float64)
        rg_off[1, 2] = 3.5      # column 2 -> secondary range 5.5:
        #                         sub-swath int(6.0) = 6 -> sub-swath 2,
        #                         exception round(5.5) = 6 (half-even)
        rg_off[1, 4] = -1.5     # column 4 -> secondary range 2.5:
        #                         sub-swath int(3.0) = 3 -> sub-swath 1,
        #                         exception round(2.5) = 2
        rg_off[1, 6] = -6.7     # column 6 -> secondary range -0.7:
        #                         int(-0.2) truncates toward zero to 0
        #                         (in bounds, sub-swath 1) while
        #                         exception round(-0.7) = -1 is out of
        #                         bounds and drops the secondary bits
        az_off = np.zeros((lines, samples), dtype=np.float64)
        rg_off_path = tmp_path / "rounding_range.off"
        az_off_path = tmp_path / "rounding_azimuth.off"
        make_offset_raster(rg_off_path, rg_off)
        make_offset_raster(az_off_path, az_off)

        idx_azi = np.arange(lines, dtype=np.float64)
        idx_rg = np.arange(samples, dtype=np.float64)
        mask = unpack_mask(generate_insar_mask(
            FakeSLC(swath), FakeSLC(swath), ref_h5, sec_h5,
            str(rg_off_path), str(az_off_path), "A", idx_azi, idx_rg))
        ref_h5.close()
        sec_h5.close()

        # column 2: ref sub-swath 1, sec sub-swath 2, sec exception
        # byte from round-half-even column 6 (zero)
        assert mask[1, 2] == 12
        # column 4: ref sub-swath 2, sec sub-swath 1, sec exception
        # byte 0x11 from round-half-even column 2
        assert mask[1, 4] == (0x11 << 8) | 21
        # column 6: ref sub-swath 2, sec column int(-0.2) = 0 ->
        # sub-swath 1 (a floor would give -1 -> 0), no exception bits
        assert mask[1, 6] == 21
        # zero-offset columns keep matching digits and no exception bits
        assert mask[1, 0] == 11
        assert mask[1, 5] == 22
