from dataclasses import dataclass

import journal
import numpy as np
from isce3.io import compute_page_size, optimize_chunk_size
from nisar.workflows.helpers import get_cfg_freq_pols

from .utils import (compute_number_of_elements,
                    get_geolocation_grid_cube_shape,
                    get_interferogram_dataset_shape,
                    get_pixel_offsets_dataset_shape, get_radar_grid_cube_shape,
                    get_unwrapped_interferogram_dataset_shape)

# Float 32 data type size
FLOAT32_SIZE = np.dtype("float32").itemsize

# Complex 64 data type size
COMPLEX64_SIZE = np.dtype("complex64").itemsize

def _get_page_size(chunk_sz, ds_shape, ds_type_sz):
    """
    Helper function to compute the page size in bytes
    """

    ds_sz = optimize_chunk_size(chunk_sz, ds_shape)
    mem_footprint = compute_number_of_elements(ds_sz) * ds_type_sz

    return compute_page_size(mem_footprint)

@dataclass
class HDF5OutputOptimizedConfig:
    """
    A data class describing the HDF5 optimizer configurations
    for InSAR products

    Attributes
    ----------
    page_enabled : bool
        Flag to enable/disable paging
    chunk_size : tuple[int, int]
        Chunk size of the dataset along rows and columns
    page_size : float
        The page size to create the InSAR HDF5 products
    compression_enabled: bool
        Flag to enable/disable data compression
    compression_type: str
        Data compression algorithm (default: gzip)
    compression_level: int
        Level of data compression (1: low compression, 9: high compression)
    shuffle_filter: bool
        Flag to enable/disable shuffle filter
    """
    page_enabled: bool
    chunk_size: tuple
    page_size: float
    compression_enabled: bool
    compression_type: str
    compression_level: int
    shuffle_filter: bool

    @classmethod
    def make_base(cls, cfg : dict):
        """
        HDF5 Configuration Constructor of InSAR base product

        Parameters
        ----------
        cfg : dict
            InSAR runconfig dictionary
        """
        page_enabled = cfg['output']['page_enabled']
        compression_enabled = cfg['output']['compression_enabled']
        compression_level = cfg['output']['compression_level']
        chunk_size = cfg['output']['chunk_size']
        shuffle_filter = cfg['output']['shuffle']

        # If the compression is enabled, we need the chunking
        if compression_enabled and (chunk_size is None):
            error_channel = journal.error(
                "nisar.products.insar.InSAR_HDF5_optimizer_config.Base")
            err_msg = 'compressing the hdf5 file needs chunking'
            error_channel.log(err_msg)
            raise ValueError(err_msg)

        page_size = compute_page_size(
            compute_number_of_elements(chunk_size) * FLOAT32_SIZE)

        return cls(page_enabled,
                   chunk_size,
                   page_size,
                   compression_enabled,
                   'gzip',
                   compression_level,
                   shuffle_filter)

    @classmethod
    def make_rifg(cls,  cfg : dict):
        """
        HDF5 Configuration Constructor of InSAR RIFG product

        Parameters
        ----------
        cfg : dict
            InSAR runconfig dictionary
        """
        base = cls.make_base(cfg)
        if base.chunk_size is None:
            return base

        # Compute the datacube page size
        geolocation_grid_shape = get_geolocation_grid_cube_shape(cfg)
        page_size = _get_page_size((1,
                                    base.chunk_size[0],
                                    base.chunk_size[1]),
                                   geolocation_grid_shape, FLOAT32_SIZE)

        # Compute the largest page_size against the dataset
        for freq, *_ in get_cfg_freq_pols(cfg):
            ifgram_shape = get_interferogram_dataset_shape(cfg, freq)
            ifgram_footprint = _get_page_size(base.chunk_size,
                                              ifgram_shape,
                                              COMPLEX64_SIZE)

            off_shape = get_pixel_offsets_dataset_shape(cfg, freq)
            off_footprint = _get_page_size(base.chunk_size,
                                           off_shape,
                                           FLOAT32_SIZE)

            page_size = max(ifgram_footprint, off_footprint, page_size)

        page_size = compute_page_size(page_size)

        return cls(base.page_enabled,
                   base.chunk_size,
                   page_size,
                   base.compression_enabled,
                   base.compression_type,
                   base.compression_level,
                   base.shuffle_filter)

    @classmethod
    def make_roff(cls, cfg: dict):
        """
        HDF5 Configuration Constructor of InSAR ROFF product

        Parameters
        ----------
        cfg : dict
            InSAR runconfig dictionary
        """

        base = cls.make_base(cfg)
        if base.chunk_size is None:
            return base

        # Compute the datacube page size
        geolocation_grid_shape = get_geolocation_grid_cube_shape(cfg)
        page_size = _get_page_size((1,
                                    base.chunk_size[0],
                                    base.chunk_size[1]),
                                    geolocation_grid_shape,
                                    FLOAT32_SIZE)

        # Compute the largest page_size against the dataset
        for freq, *_ in get_cfg_freq_pols(cfg):
            off_shape = get_pixel_offsets_dataset_shape(cfg, freq)
            footprint = _get_page_size(base.chunk_size,
                                       off_shape,
                                       FLOAT32_SIZE)

            page_size = max(footprint, page_size)

        return cls(base.page_enabled,
                   base.chunk_size,
                   page_size,
                   base.compression_enabled,
                   base.compression_type,
                   base.compression_level,
                   base.shuffle_filter)

    @classmethod
    def make_runw(cls, cfg : dict):
        """
        HDF5 Configuration Constructor of InSAR RUNW product

        Parameters
        ----------
        cfg : dict
            InSAR runconfig dictionary
        """
        base = cls.make_base(cfg)
        if base.chunk_size is None:
            return base

        # Compute the datacube page size
        geolocation_grid_shape = get_geolocation_grid_cube_shape(cfg)
        page_size = _get_page_size((1,
                                    base.chunk_size[0],
                                    base.chunk_size[1]),
                                    geolocation_grid_shape,
                                    FLOAT32_SIZE)

        # Compute the largest page_size against the dataset
        for freq, *_ in get_cfg_freq_pols(cfg):
            ifgram_shape = get_unwrapped_interferogram_dataset_shape(cfg, freq)
            ifgram_footprint = _get_page_size(base.chunk_size,
                                              ifgram_shape,
                                              FLOAT32_SIZE)

            off_shape = get_pixel_offsets_dataset_shape(cfg, freq)
            off_footprint = _get_page_size(base.chunk_size,
                                           off_shape,
                                           FLOAT32_SIZE)

            page_size = max(ifgram_footprint, off_footprint, page_size)

        return cls(base.page_enabled,
                   base.chunk_size,
                   page_size,
                   base.compression_enabled,
                   base.compression_type,
                   base.compression_level,
                   base.shuffle_filter)

    @classmethod
    def make_goff(cls, cfg: dict):
        """
        HDF5 Configuration Constructor of InSAR GOFF product

        Parameters
        ----------
        cfg : dict
            InSAR runconfig dictionary
        """
        base = cls.make_base(cfg)
        if base.chunk_size is None:
            return base

        # Compute the datacube page size
        radar_grid_shape = get_radar_grid_cube_shape(cfg)
        page_size = _get_page_size((1,
                                    base.chunk_size[0],
                                    base.chunk_size[1]),
                                    radar_grid_shape,
                                    FLOAT32_SIZE)

        # Compute the largest footprint to compute the page size
        proc_cfg = cfg["processing"]
        geogrids = proc_cfg["geocode"]["geogrids"]
        for freq, *_ in get_cfg_freq_pols(cfg):
            goff_geogrids = geogrids[freq]
            goff_shape = (goff_geogrids.length,goff_geogrids.width)
            footprint = _get_page_size(base.chunk_size,
                                       goff_shape,
                                       FLOAT32_SIZE)

            page_size = max(footprint, page_size)

        return cls(base.page_enabled,
                   base.chunk_size,
                   page_size,
                   base.compression_enabled,
                   base.compression_type,
                   base.compression_level,
                   base.shuffle_filter)

    @classmethod
    def make_gunw(cls, cfg : dict):
        """
        HDF5 Configuration Constructor of InSAR GUNW product

        Parameters
        ----------
        cfg : dict
            InSAR runconfig dictionary
        """
        base = cls.make_base(cfg)
        if base.chunk_size is None:
            return base

        # Compute the datacube page size
        radar_grid_shape = get_radar_grid_cube_shape(cfg)
        page_size = _get_page_size((1,
                                    base.chunk_size[0],
                                    base.chunk_size[1]),
                                    radar_grid_shape,
                                    FLOAT32_SIZE)

        # Compute the largest footprint to compute the page size
        proc_cfg = cfg["processing"]
        wrapped_igram_geogrids = proc_cfg["geocode"]["wrapped_igram_geogrids"]
        geogrids = proc_cfg["geocode"]["geogrids"]
        for freq, *_ in get_cfg_freq_pols(cfg):
            gunw_geogrids = wrapped_igram_geogrids[freq]
            gunw_shape = (gunw_geogrids.length,gunw_geogrids.width)
            wrapped_igram_footprint = _get_page_size(base.chunk_size,
                                                     gunw_shape,
                                                     COMPLEX64_SIZE)

            off_geogrids = geogrids[freq]
            off_shape = (off_geogrids.length,off_geogrids.width)
            off_footprint = _get_page_size(base.chunk_size,
                                           off_shape,
                                           FLOAT32_SIZE)

            page_size = max(wrapped_igram_footprint, off_footprint,
                            page_size)

        return cls(base.page_enabled,
                   base.chunk_size,
                   page_size,
                   base.compression_enabled,
                   base.compression_type,
                   base.compression_level,
                   base.shuffle_filter)


def get_InSAR_output_options(kwds: dict, product_name: str):
    '''
    Get InSAR product specific options for cloud optimization and compression.

    Parameters
    ----------
    kwds: dict
        Keyword arguments for opening h5py.File in write mode
        without cloud optimization options
    product_name: str
        InSAR product name (e.g. 'RIFG', 'RUNW', 'ROFF', 'GOFF', 'GUNW')

    Returns
    -------
    hd5_opt_config: HDF5OutputOptimizedConfig
        InSAR product specific cloud optimization HDF5 parameters
    kwds: dict
        Keyword arguments for opening h5py.File in write mode with cloud optimization options
    '''

    # Deepcopy the kwds to avoid the in-place
    # change of the kwds, we use loop here instead of copy.deepcopy()
    # is because the the copy.deepcopy()
    # cannot pickle 'isce3.ext.isce3.product.GeoGridParameters' object
    new_kwds = {}
    for key, val in kwds.items():
        new_kwds[key] = val

    try:
        factory = getattr(HDF5OutputOptimizedConfig,
                          f"make_{product_name.lower()}")
    except AttributeError:
        error_channel = journal.error(
            "nisar.products.insar.InSAR_HDF5_optimizer_config."
            "get_InSAR_output_options")
        err_str = f"{product_name} is not a valid InSAR product"
        error_channel.log(err_str)
        raise ValueError(err_str)

    hdf5_opt_config = factory(new_kwds['runconfig_dict'])

    if hdf5_opt_config.page_enabled:
        hdf5_output_config = {'fs_strategy':'page',
                            'fs_page_size':hdf5_opt_config.page_size,
                            'fs_persist':True}

        for key, val in hdf5_output_config.items():
            if key not in new_kwds:
                new_kwds[key] = val

    return hdf5_opt_config, new_kwds
