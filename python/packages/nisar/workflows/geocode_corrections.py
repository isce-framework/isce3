'''
Compute azimuth and slant range geocoding corrections as LUT2d
'''
import isce3
from isce3.atmosphere.tec_product import tec_lut2d_from_json_srg, tec_lut2d_from_json_az
from nisar.products.readers import SLC
import journal


def _get_accumulated_azimuth_corrections(cfg, slc, frequency, orbit):
    '''
    Compute, accumulate, and azimuth geolocation corrections as LUT2d. Default
    to default LUT2d if provided parameters do not require corrections to be
    computed.

    Parameters
    ----------
    cfg: dict
        Dict containing the runconfiguration parameters
    slc: nisar.products.readers.SLC
        NISAR single look complex (SLC) object containing swath and radar grid
        parameters
    frequency: ['A', 'B']
        Str identifcation for NISAR SLC frequencies
    orbit: isce3.core.Orbit
        Object containing orbit associated with SLC

    Returns
    -------
    tec_correction: isce3.core.LUT2d
        Azimuth correction for geocoding. Currently only TEC corrections
        are considered. If no TEC JSON file is provided in the cfg parameter,
        an empty isce3.core.LUT2d will be returned.
    '''
    # Compute TEC slant range correction if TEC file is provided
    tec_file = cfg["dynamic_ancillary_file_group"]['tec_file']

    # Empty LUT2d as default azimuth geolocation correction LUT
    tec_correction = isce3.core.LUT2d()

    # Ionosphere
    if tec_file is not None:
        center_freq = slc.getSwathMetadata(frequency).processed_center_frequency
        radar_grid = slc.getRadarGrid(frequency)

        tec_correction = tec_lut2d_from_json_az(tec_file, center_freq, orbit,
                                                radar_grid)

    return tec_correction


def _get_accumulated_srange_corrections(cfg, slc, frequency, orbit):
    '''
    Compute, accumulate, and return slant range corrections as LUT2d. Default
    to default LUT2d if provided parameters do not require corrections to be
    computed.

    Currently on TEC corrections available. Others will be added as they
    become available.

    Parameters
    ----------
    cfg: dict
        Dict containing the runconfiguration parameters
    slc: nisar.products.readers.SLC
        NISAR single look complex (SLC) object containing swath and radar grid
        parameters
    frequency: ['A', 'B']
        Str identifcation for NISAR SLC frequencies
    orbit: isce3.core.Orbit
        Object containing orbit associated with SLC

    Yields
    ------
    tec_correction: isce3.core.LUT2d
        Slant range correction for geocoding. Currently only TEC corrections
        are considered. If no TEC JSON file is provided in the cfg parameter,
        a default isce3.core.LUT2d will be passed back.
    '''
    # Compute TEC slant range correction if TEC file is provided
    tec_file = cfg["dynamic_ancillary_file_group"]['tec_file']

    # Empty LUT2d as default slant range geolocation correction LUT
    tec_correction = isce3.core.LUT2d()

    # Ionosphere
    if tec_file is not None:
        center_freq = slc.getSwathMetadata(frequency).processed_center_frequency
        doppler = isce3.core.LUT2d()
        radar_grid = slc.getRadarGrid(frequency)

        # DEM file for DEM interpolator and ESPF for ellipsoid
        dem_file = cfg['dynamic_ancillary_file_group']['dem_file']

        tec_correction = tec_lut2d_from_json_srg(tec_file, center_freq, orbit,
                                                 radar_grid, doppler, dem_file)

    return tec_correction


def get_az_srg_corrections(cfg, slc, frequency, orbit):
    '''
    Compute azimuth and slant range geocoding corrections and return as LUT2d.
    Default to default LUT2d for either if provided parameters do not require
    corrections to be computed.

    Parameters
    ----------
    cfg: dict
        Dict containing the runconfiguration parameters
    slc: nisar.products.readers.SLC
        NISAR single look complex (SLC) object containing swath and radar grid
        parameters
    frequency: ['A', 'B']
        Str identifcation for NISAR SLC frequencies
    orbit: isce3.core.Orbit
        Object containing orbit associated with SLC

    Yields
    ------
    az_corrections: isce3.core.LUT2d
        Azimuth correction for geocoding. Currently only no corrections are
        computed and a default isce3.core.LUT2d is be passed back.
    srange_corrections: isce3.core.LUT2d
        Slant range correction for geocoding. Currently only TEC corrections
        are considered. If no TEC JSON file is provided in the cfg parameter,
        a default isce3.core.LUT2d will be passed back.
    '''
    az_corrections = _get_accumulated_azimuth_corrections(cfg, slc,
                                                          frequency, orbit)
    srange_corrections = _get_accumulated_srange_corrections(cfg, slc,
                                                             frequency, orbit)

    return az_corrections, srange_corrections


def get_offset_lut(cfg, slc, frequency, orbit):
    '''
    A placeholder to compute timing correction based on offset tracking (ampcor)

    Parameters
    ----------
    cfg: dict
    frequency: ['A', 'B']
        Str identifcation for NISAR SLC frequencies
    slc: nisar.products.readers.SLC
        NISAR single look complex (SLC) object containing swath and radar grid
        parameters
    orbit: isce3.core.Orbit
        Object containing orbit associated with SLC

    Returns
    -------
    az_lut: isce3.core.LUT2d
        2d LUT in azimuth time for geolocation correction in azimuth direction.
    rg_lut: isce3.core.LUT2d
        2d LUT in meters for geolocation correction in slant range
    '''
    info_channel = journal.info("geocode_corrections.get_offset_lut")

    info_channel.log('Data-driven GSLC will be implemented in the next release.'
                     ' Currently returning empty LUT2d of timing corrections in'
                     ' both range and azimuth directions.')

    rg_lut = isce3.core.LUT2d()
    az_lut = isce3.core.LUT2d()
    return az_lut, rg_lut
