import h5py
import numpy as np
from nisar.workflows.h5_prep import set_get_geo_info
from nisar.workflows.helpers import get_cfg_freq_pols

from .InSAR_base_writer import InSARBaseWriter
from .InSAR_L2_writer import L2InSARWriter
from .InSAR_products_info import InSARProductsInfo
from .product_paths import GUNWGroupsPaths
from .RIFG_writer import RIFGWriter
from .RUNW_writer import RUNWWriter
from .units import Units


class GUNWWriter(RUNWWriter, RIFGWriter, L2InSARWriter):
    """
    Writer class for GUNW product inherent from both the RUNWWriter,
    RIFGWriter, and the L2InSARWriter
    """
    def __init__(self, **kwds):
        """
        Constructor for GUNW writer class
        """
        super().__init__(**kwds)

        # group paths are GUNW group paths
        self.group_paths = GUNWGroupsPaths()

        # GUNW product information
        self.product_info = InSARProductsInfo.GUNW()

    def save_to_hdf5(self):
        """
        Save to HDF5
        """
        L2InSARWriter.save_to_hdf5(self)

    def add_root_attrs(self):
        """
        add root attributes
        """
        InSARBaseWriter.add_root_attrs(self)

        self.attrs["title"] = np.string_("NISAR L2 GUNW Product")
        self.attrs["reference_document"] = \
            np.string_("D-102272 NISAR NASA SDS Product Specification"
                       " L2 Geocoded Unwrapped Interferogram")

        ctype = h5py.h5t.py_create(np.complex64)
        ctype.commit(self["/"].id, np.string_("complex64"))

    def add_radar_grid_cubes(self):
        """
        Add the radar grid cubes
        """
        L2InSARWriter.add_radar_grid_cubes(self)

        ## Add the radar grid cubes of solid eath tide phase for along-track and along-slant range.
        proc_cfg = self.cfg["processing"]
        tropo_cfg = proc_cfg['troposphere_delay']
        radar_grid_cubes_geogrid = proc_cfg["radar_grid_cubes"]["geogrid"]
        radar_grid_cubes_heights = proc_cfg["radar_grid_cubes"]["heights"]

        radar_grid = self[self.group_paths.RadarGridPath]
        descrs = ["Solid Earth tides phase along slant range direction",
                  'Solid Earth tides phase in along-track direction']
        product_names = ['slantRangeSolidEarthTidesPhase']

        # Add the troposphere datasets to the radarGrid cube
        if tropo_cfg['enabled']:
            for delay_type in ['wet', 'hydrostatic', 'comb']:
                if tropo_cfg[f'enable_{delay_type}_product']:
                    descrs.append(f"{delay_type.capitalize()} component "
                                  "of the troposphere phase screen")
                    product_names.append(f'{delay_type}TroposphericPhaseScreen')

        cube_shape = [len(radar_grid_cubes_heights),
                      radar_grid_cubes_geogrid.length,
                      radar_grid_cubes_geogrid.width]

        # Retrieve the x, y, and z coordinates from the radargrid cube
        # Since the radargrid cube has been added, it is safe to
        # access those coordinates here.
        xds = radar_grid['xCoordinates']
        yds = radar_grid['yCoordinates']
        zds = radar_grid['heightAboveEllipsoid']

        for product_name, descr in zip(product_names,descrs):
            if product_name not in radar_grid:
                ds = radar_grid.require_dataset(name=product_name,
                                                shape=cube_shape,
                                                dtype=np.float64)
                ds.attrs['description'] = np.string_(descr)
                ds.attrs['units'] = Units.radian
                ds.attrs['grid_mapping'] = np.string_('projection')
                ds.dims[0].attach_scale(zds)
                ds.dims[1].attach_scale(yds)
                ds.dims[2].attach_scale(xds)

    def add_algorithms_to_procinfo_group(self):
        """
        Add the algorithms to processingInformation group
        """
        RUNWWriter.add_algorithms_to_procinfo_group(self)
        L2InSARWriter.add_geocoding_to_algo_group(self)

    def add_parameters_to_procinfo_group(self):
        """
        Add parameters group to processingInformation/parameters group
        """
        RUNWWriter.add_parameters_to_procinfo_group(self)

        # the unwrappedInterfergram group under the processingInformation/parameters
        # group is copied from the RUNW product, but the name in RUNW product is
        # 'interferogram', while in GUNW its name is 'unwrappedInterferogram'. Here
        # is to rename the interfegram group name to unwrappedInterferogram group name
        old_igram_group_name = \
            f"{self.group_paths.ParametersPath}/interferogram"
        new_igram_group_name = \
            f"{self.group_paths.ParametersPath}/unwrappedInterferogram"
        self.move(old_igram_group_name, new_igram_group_name)

        # the wrappedInterfergram group under the processingInformation/parameters
        # group is copied from the RIFG product, but the name in RIFG product is
        # 'interferogram', while in GUNW its name is 'wrappedInterferogram'. Here
        # is to rename the interfegram group name to wrappedInterferogram group name
        RIFGWriter.add_interferogram_to_procinfo_params_group(self)
        new_igram_group_name = \
            f"{self.group_paths.ParametersPath}/wrappedInterferogram"
        self.move(old_igram_group_name, new_igram_group_name)

        L2InSARWriter.add_geocoding_to_procinfo_params_group(self)

        # Update the descriptions of the reference and secondary
        for rslc_name in ['reference', 'secondary']:
            rslc = self[self.group_paths.ParametersPath][rslc_name]
            rslc['referenceTerrainHeight'].attrs['description'] = \
                np.string_("Reference Terrain Height as a function of"
                           f" map coordinates for {rslc_name} RSLC")
            rslc['referenceTerrainHeight'].attrs['units'] = \
                Units.meter

    def add_grids_to_hdf5(self):
        """
        Add grids to HDF5
        """
        L2InSARWriter.add_grids_to_hdf5(self)

        pcfg = self.cfg["processing"]
        geogrids = pcfg["geocode"]["geogrids"]
        wrapped_igram_geogrids = pcfg["geocode"]["wrapped_igram_geogrids"]

        grids_val = np.string_("projection")

        # Only add the common fields such as list of polarizations, pixel offsets, and center frequency
        for freq, pol_list, _ in get_cfg_freq_pols(self.cfg):
            # Create the swath group
            grids_freq_group_name = (
                f"{self.group_paths.GridsPath}/frequency{freq}"
            )
            grids_freq_group = self.require_group(grids_freq_group_name)

            # Create the pixeloffsets group
            offset_group_name = f"{grids_freq_group_name}/pixelOffsets"
            self.require_group(offset_group_name)

            rslc_freq_group = self.ref_h5py_file_obj[
                f"{self.ref_rslc.SwathPath}/frequency{freq}"
            ]

            rslc_freq_group.copy("numberOfSubSwaths",
                                 grids_freq_group)

            unwrapped_geogrids = geogrids[freq]
            wrapped_geogrids = wrapped_igram_geogrids[freq]

            # shape of the unwrapped phase
            unwrapped_shape = (
                unwrapped_geogrids.length,
                unwrapped_geogrids.width,
            )

            # shape of the wrapped interferogram
            wrapped_shape = (
                wrapped_geogrids.length,
                wrapped_geogrids.width,
            )

            unwrapped_group_name = \
                f"{grids_freq_group_name}/unwrappedInterferogram"

            wrapped_group_name = \
                f"{grids_freq_group_name}/wrappedInterferogram"

            pixeloffsets_group_name = \
                f"{grids_freq_group_name}/pixelOffsets"

            unwrapped_group = self.require_group(unwrapped_group_name)

            # set the geo information for the mask
            yds, xds = set_get_geo_info(
                self,
                unwrapped_group_name,
                unwrapped_geogrids,
            )

            # Create mask only if layover shadow mask is created
            # or if we have a water mask assigned from runconfig
            if pcfg['rdr2geo']['write_layover_shadow'] or \
                    self.cfg['dynamic_ancillary_file_group']['water_mask_file'] is not None:
                self._create_2d_dataset(
                    unwrapped_group,
                    "mask",
                    unwrapped_shape,
                    np.uint8,
                    "Byte layer with flags for various channels"
                    " (e.g. layover/shadow, data quality)"
                    ,
                    Units.dn,
                    grids_val,
                    xds=xds,
                    yds=yds,
                    compression_enabled=self.cfg['output']['compression_enabled'],
                    compression_level=self.cfg['output']['compression_level'],
                    chunk_size=self.cfg['output']['chunk_size'],
                    shuffle_filter=self.cfg['output']['shuffle']
                )

            for pol in pol_list:
                unwrapped_pol_name = f"{unwrapped_group_name}/{pol}"
                unwrapped_pol_group = self.require_group(unwrapped_pol_name)

                yds, xds = set_get_geo_info(
                    self,
                    unwrapped_pol_name,
                    unwrapped_geogrids,
                )

                #unwrapped dataset parameters as tuples in the following
                #order: dataset name, data type, description, and units
                unwrapped_ds_params = [
                    ("coherenceMagnitude", np.float32,
                     f"Coherence magnitude between {pol} layers",
                     Units.unitless),
                    ("connectedComponents", np.uint16,
                     f"Connected components for {pol} layer",
                     Units.unitless),
                    ("ionospherePhaseScreen", np.float32,
                     "Ionosphere phase screen",
                     Units.radian),
                    ("ionospherePhaseScreenUncertainty", np.float32,
                     "Uncertainty of the ionosphere phase screen",
                     "radians"),
                    ("unwrappedPhase", np.float32,
                    f"Unwrapped interferogram between {pol} layers",
                     Units.radian),
                ]

                for ds_param in unwrapped_ds_params:
                    ds_name, ds_datatype, ds_description, ds_unit\
                        = ds_param
                    self._create_2d_dataset(
                        unwrapped_pol_group,
                        ds_name,
                        unwrapped_shape,
                        ds_datatype,
                        ds_description,
                        ds_unit,
                        grids_val,
                        xds=xds,
                        yds=yds,
                        compression_enabled=self.cfg['output']['compression_enabled'],
                        compression_level=self.cfg['output']['compression_level'],
                        chunk_size=self.cfg['output']['chunk_size'],
                        shuffle_filter=self.cfg['output']['shuffle']
                    )

                wrapped_pol_name = f"{wrapped_group_name}/{pol}"
                wrapped_pol_group = self.require_group(wrapped_pol_name)

                yds, xds = set_get_geo_info(
                    self,
                    wrapped_pol_name,
                    wrapped_geogrids,
                )

                #wrapped dataset parameters as tuples in the following
                #order: the dataset name,data type, description, and units
                wrapped_ds_params = [
                    ("coherenceMagnitude", np.float32,
                     f"Coherence magnitude between {pol} layers",
                     Units.unitless),
                    ("wrappedInterferogram", np.complex64,
                     f"Complex wrapped interferogram between {pol} layers",
                     Units.unitless),
                ]

                for ds_param in wrapped_ds_params:
                    ds_name, ds_datatype, ds_description, ds_unit\
                        = ds_param
                    self._create_2d_dataset(
                        wrapped_pol_group,
                        ds_name,
                        wrapped_shape,
                        ds_datatype,
                        ds_description,
                        ds_unit,
                        grids_val,
                        xds=xds,
                        yds=yds,
                        compression_enabled=self.cfg['output']['compression_enabled'],
                        compression_level=self.cfg['output']['compression_level'],
                        chunk_size=self.cfg['output']['chunk_size'],
                        shuffle_filter=self.cfg['output']['shuffle']
                    )

                pixeloffsets_pol_name = f"{pixeloffsets_group_name}/{pol}"
                pixeloffsets_pol_group = self.require_group(
                    pixeloffsets_pol_name
                )

                yds, xds = set_get_geo_info(
                    self,
                    pixeloffsets_pol_name,
                    unwrapped_geogrids,
                )

                # pixel offsets dataset parameters as tuples in the following
                # order: dataset name,data type, description, and units
                pixel_offsets_ds_params = [
                    ("alongTrackOffset", np.float32,
                     "Along-track offset",
                     Units.meter),
                    ("correlationSurfacePeak", np.float32,
                     "Normalized cross-correlation surface peak",
                     Units.unitless),
                    ("slantRangeOffset", np.float32,
                     "Slant range offset",
                     Units.meter),
                ]

                for ds_param in pixel_offsets_ds_params:
                    ds_name, ds_datatype, ds_description, ds_unit\
                        = ds_param
                    self._create_2d_dataset(
                        pixeloffsets_pol_group,
                        ds_name,
                        unwrapped_shape,
                        ds_datatype,
                        ds_description,
                        ds_unit,
                        grids_val,
                        xds=xds,
                        yds=yds,
                        compression_enabled=self.cfg['output']['compression_enabled'],
                        compression_level=self.cfg['output']['compression_level'],
                        chunk_size=self.cfg['output']['chunk_size'],
                        shuffle_filter=self.cfg['output']['shuffle']
                    )