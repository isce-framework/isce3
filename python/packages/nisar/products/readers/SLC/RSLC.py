# -*- coding: utf-8 -*-
from __future__ import annotations

import h5py
import journal
import logging
import pyre
import re
import numpy as np

from nisar.noise import NoiseEquivalentBackscatterProduct
from isce3.core import DateTime
from isce3.core.types import ComplexFloat16Decoder, is_complex32

from .SLCBase import SLCBase

PRODUCT = 'RSLC'

log = logging.getLogger("nisar.products.readers.RSLC")


class RSLC(SLCBase, family='nisar.productreader.rslc'):
    """
    Class for parsing NISAR RSLC products into ISCE3 structures.
    """

    productValidationType = pyre.properties.str(default=PRODUCT)
    productValidationType.doc = 'Validation tag to ensure correct product type'

    _ProductType = pyre.properties.str(default=PRODUCT)
    _ProductType.doc = 'The type of the product.'

    @property
    def ProductPath(self) -> str:
        # The product group name should be "RSLC" per the spec. However, early
        # sample products used "SLC" instead, and identification.productType is
        # not reliable, either. We maintain compatibility with both options.
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            g = f[self.RootPath]
            if "RSLC" in g:
                return f"{g.name}/RSLC"
            elif "SLC" in g:
                return f"{g.name}/SLC"
        raise RuntimeError("HDF5 file missing 'RSLC' or 'SLC' product group.")

    def imageDatasetPath(self, frequency: str, polarization: str) -> str:
        # implementation of GenericProduct.imageDatasetPath
        data_path = f"{self.SwathPath}/frequency{frequency}/{polarization}"
        return data_path

    def getSlcDatasetAsNativeComplex(
        self,
        frequency: str,
        polarization: str,
    ) -> h5py.Dataset | ComplexFloat16Decoder:
        """
        Get an SLC raster layer as a complex64 or complex128 dataset.

        Return the SLC dataset corresponding to a given frequency sub-band and
        polarization from the HDF5 file as a complex64 (i.e. pairs of 32-bit floats)
        or complex128 (i.e. pairs of 64-bit floats) dataset. If the data was stored as
        complex32 (i.e. pairs of 16-bit floats), it will be lazily converted to
        complex64 when accessed.

        Parameters
        ----------
        frequency : "A" or "B"
            The frequency letter, either "A" or "B".
        polarization: str
            The polarization term associated with the data array.
            One of "HH", "HV", "VH", "VV", "LH", "LV", "RH", "RV".

        Returns
        -------
        h5py.Dataset or isce3.core.types.ComplexFloat16Decoder
            The HDF5 dataset, possibly wrapped in a decoder layer that handles
            converting from half precision complex values to single precision.
        """
        dataset = self.getSlcDataset(frequency, polarization)

        if is_complex32(dataset):
            return ComplexFloat16Decoder(dataset)
        else:
            return dataset

    def getProductLevel(self):
        """
        Returns the product level
        """
        return "L1"

    def is_dataset_complex32(self, freq: str, pol: str) -> bool:
        """
        Determine if RSLC raster is of data type complex32

        Parameters
        ----------
        freq : "A" or "B"
            The frequency letter, either "A" or "B".
        pol: str
            The polarization term associated with the data array.
            One of "HH", "HV", "VH", "VV", "LH", "LV", "RH", "RV".
        """
        # Set error channel
        error_channel = journal.error('SLC.is_dataset_complex32')

        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as h:
            freq_path = f'/{self.SwathPath}/frequency{freq}'
            if freq_path not in h:
                err_str = f'Frequency {freq} not found in SLC'
                error_channel.log(err_str)
                raise LookupError(err_str)

            slc_path = self.slcPath(freq, pol)
            if slc_path not in h:
                err_str = f'Polarization {pol} for frequency {freq} not found in SLC'
                error_channel.log(err_str)
                raise LookupError(err_str)

            return is_complex32(h[slc_path])

    def getNoiseEquivalentBackscatter(self, frequency=None, pol=None):
        '''
        Extract noise equivalent backscatter product for a particular
        frequency band and TxRx polarization.  It's conceptually the same as
        as noise equivalent sigma zero (NESZ) but agnostic with respect to the
        area normalization convention.

        Parameters
        ----------
        frequency : str, optional
            Frequency band such as 'A', 'B'.
            Default is the very first one in lexicographical order.
        pol : str, optional
            TxRx polarization such as 'HH', 'HV', etc.
            Default is the first co-pol in frequency if `frequency`
            otherwise the very first co-pol in very first frequency
            band. If no co-pol, the first cross-pol product will
            be picked.

        Returns
        -------
        nisar.noise.NoiseEquivalentBackscatterProduct

        '''
        # set frequency and pol
        if frequency is None:
            frequency = self._getFirstFrequency()
        if pol is None:
            pols = self.polarizations[frequency]
            co_pol = [p for p in pols if p[0] == p[1] or p[0] in ('L', 'R')]
            if len(co_pol) == 0:  # no co-pol
                pol = pols[0]
            else:  # there exists a co-pol
                pol = co_pol[0]

        # Save typing...
        cal, freq = self.CalibrationInformationPath, f'frequency{frequency}'

        # Set paths relative to cal group. Support three product spec versions.
        # Keys correspond to the first tag of the NISAR PIX repo that implements
        # the given data layout (though v0.0.0 doesn't exist and is just
        # shorthand for the first ever version).
        layouts = {
            "v1.2.0": {
                "noise": _h5join(cal, freq, "noiseEquivalentBackscatter", pol),
                "time": _h5join(cal, freq, "noiseEquivalentBackscatter",
                    "zeroDopplerTime"),
                "range": _h5join(cal, freq, "noiseEquivalentBackscatter",
                    "slantRange"),
            },
            "v1.0.0": {
                "noise": _h5join(cal, freq, "nes0", pol),
                "time": _h5join(cal, freq, "nes0", "zeroDopplerTime"),
                "range": _h5join(cal, freq, "nes0", "slantRange"),
            },
            "v0.0.0": {
                "noise": _h5join(cal, freq, pol, "nes0"),
                "time": _h5join(cal, "zeroDopplerTime"),
                "range": _h5join(cal, "slantRange"),
            },
        }

        # parse all fields for NESZ
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as fid:
            for version, paths in layouts.items():
                if paths["noise"] in fid:
                    break
                log.warning("Couldn't find noise with RSLC schema "
                    f"corresponding to tag {version} of NISAR_PIX.")
            else:
                raise IOError("Could not find noise layer in RSLC file.")

            noise = fid[paths["noise"]][:]
            sr = fid[paths["range"]][:]
            azt_dset = fid[paths["time"]]
            azt = azt_dset[:]
            units = azt_dset.attrs['units']
            # This attribute should be `bytes`, but may be stored as `str` in nonconforming legacy NISAR products.
            if not isinstance(units, str):
                units = units.decode()

        # datetime UTC pattern to look for in units to get epoch
        dt_pat = re.compile(
            '[0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]{2}:[0-9]{2}:[0-9]{2}(?:[.][0-9]{0,9})?'
        )
        matches = dt_pat.findall(units)
        if len(matches) != 1:
            raise RuntimeError(
                f"missing epoch in zeroDopplerTime units attribute: {units!r}"
            )
        utc_str = matches[0]
        epoch = DateTime(utc_str)
        # build and return noise product
        return NoiseEquivalentBackscatterProduct(noise, sr, azt, epoch,
                                                 frequency, pol)

    def getResampledNoiseEquivalentBackscatter(
            self,
            sensing_times,
            slant_ranges=None,
            frequency=None,
            pol=None,
            range_interpolator=np.interp,
            range_interpolator_kwargs=None):
        '''
        Extract noise equivalent backscatter product for a particular
        frequency band and TxRx polarization resampled over given
        `sensing_times` and `slant_ranges`. It's conceptually the same as
        as noise equivalent sigma zero (NESZ) but agnostic with respect to the
        area normalization convention.

        Parameters
        ----------
        sensing_times: array_like
            Azimuth sensing times for the output noise product in seconds
            with respect to the reference epoch
        slant_range: array_like or None
            Slant-range distances for the output noise product in meters.
            If `None`, the slant-range distances of the noise equivalent
            backscatter look-up table (LUT) in the RSLC metadata will be
            used.
        frequency : str or None
            Frequency band such as 'A', 'B'.
            Default is the very first one in lexicographical order.
        pol : str or None
            TxRx polarization such as 'HH', 'HV', etc.
            Default is the first co-pol in frequency if `frequency`
            otherwise the very first co-pol in very first frequency
            band. If no co-pol, the first cross-pol product will
            be picked.
        range_interpolator: callable, optional
            Range 1-D interpolator. A function that uses the input `X` and `Y`
            data points to interpolate the new `Y_new` values at the `X_new`
            positions as:
            ```
            Y_new = range_interpolator(X_new, X, Y,
                                       **range_interpolator_kwargs)
            ```
            Defaults to `numpy.interp`
        range_interpolator_kwargs: dict or None
            Keyword arguments represented as a Python dictionary to be
            passed to the `range_interpolator` callable. Defaults to `None`

        Returns
        -------
        nisar.noise.NoiseEquivalentBackscatterProduct
            Resampled NoiseEquivalentBackscatterProduct

        '''

        noise_product = self.getNoiseEquivalentBackscatter(frequency=frequency,
                                                           pol=pol)
        az_orig = noise_product.az_time
        n_az_orig = len(az_orig)

        if range_interpolator_kwargs is None:
            range_interpolator_kwargs = {}
        if slant_ranges is None:
            slant_ranges = noise_product.slant_range

        # ensure that `sensing_times` and `slant_range` are numpy arrays
        sensing_times = np.asarray(sensing_times)
        slant_ranges = np.asarray(slant_ranges)

        # create array that will store the resampled noise power
        resampled_noise_power_linear = np.zeros((sensing_times.size,
                                                 slant_ranges.size),
                                                dtype=np.float64)

        # Perform nearest neighbor interpolation in azimuth and user-defined
        # interpolation (defaults to linear interpolation) along range. For
        # each azimuth coordinate in the output radar grid, find the nearest
        # azimuth coordinate in the input noise product, and interpolate
        # noise product samples along that azimuth line.

        # The array `az_times_distance` measures the distance of the new
        # azimuth times `sensing_times` with respect to the original azimuth
        # times `az_orig`. We locate the indices with minimum distances to
        # find the nearest neighbor azimuth time

        az_times_distance = np.zeros((n_az_orig, sensing_times.size),
                                     dtype=np.float64)

        for i in range(n_az_orig):
            az_times_distance[i, :] = np.absolute(sensing_times - az_orig[i])

        nearest_az_times = np.argmin(az_times_distance, axis=0)

        for i in range(n_az_orig):
            # compute the azimuth indices (lines) that will receive the
            # the current resampled line `i`. If there's no line to receive
            # the update, skip resampling, and continue to the next line `i+1`
            indices = np.where(nearest_az_times == i)[0]
            if indices.size == 0:
                continue

            new_slant_range_line = \
                range_interpolator(slant_ranges,
                                   noise_product.slant_range,
                                   noise_product.power_linear[i, :],
                                   **range_interpolator_kwargs)
            resampled_noise_power_linear[indices, :] = new_slant_range_line

        return NoiseEquivalentBackscatterProduct(
            resampled_noise_power_linear,
            np.array(slant_ranges),
            np.array(sensing_times),
            noise_product.ref_epoch,
            noise_product.freq_band,
            noise_product.txrx_pol)


def _h5join(*paths: str) -> str:
    """Join two paths to be used in HDF5"""
    # avoid repeated path separators
    return "/".join(path.rstrip("/") for path in paths)
