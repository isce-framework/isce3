#!/usr/bin/env python3

from datetime import datetime
import journal
import os
import pygrib

from isce3.io import HDF5OptimizedReader
from nisar.workflows.runconfig import RunConfig

def troposphere_delay_check(cfg):
    '''
    Check the troposphere delay yaml

    Parameters
     ----------
     cfg: dict
        configuration dictionary

     Returns
     -------
     None
    '''

    error_channel = journal.error('InsarTroposphereRunConfig.troposphere_delay_check')
    info_channel = journal.info('InsarTroposphereRunConfig.troposphere_delay_check')

    tropo_cfg = cfg['processing']['troposphere_delay']
    rslc_cfg = cfg['input_file_group']

    # only if the troposphere is enabled
    if tropo_cfg['enabled']:

        dynamic_weather_model_cfg = cfg['dynamic_ancillary_file_group']['troposphere_weather_model_files']

        for option in ['reference', 'secondary']:

            # Check the weather model file
            weather_model_file = dynamic_weather_model_cfg[f'{option}_troposphere_file']

            if (weather_model_file is None) or (not os.path.exists(weather_model_file)):
                err_str = f'{option} weather model file cannot be None or not found,' + \
                        f'please specify the {option} weather model'
                error_channel.log(err_str)
                raise ValueError(err_str)

            # Check the RSLC file
            rslc_file = rslc_cfg[f'{option}_rslc_file']

            if (rslc_file is None) or (not os.path.exists(rslc_file)):
                err_str = f'{option} RSLC file cannot be None or not found,' + \
                        f'please specify the {option} RSLC file'
                error_channel.log(err_str)
                raise ValueError(err_str)


            # RSLC start time
            with HDF5OptimizedReader(name=rslc_file, mode='r', libver='latest', swmr=True) as f:
                rslc_date = f['science/LSAR/identification/zeroDopplerStartTime'][()]\
                        .astype('datetime64[s]').astype(datetime)


            # Troposphere package
            tropo_pkg = tropo_cfg['package'].lower()

            # Weather model type
            weather_model_type = tropo_cfg['weather_model_type'].upper()

            # pyAPS only supports the grib format weather model
            if tropo_pkg == 'pyaps':

                # Check the days difference between weather model and RSLC
                grbs = pygrib.open(weather_model_file)
                if grbs is None:
                    err_str = f'{weather_model_file} is not a GRIB format'
                    error_channel.log(err_str)
                    raise ValueError(err_str)

                # Check if there are messages in the GRIB file
                if grbs.messages <= 0:
                    err_str = 'there are no messages in the GRIB file'
                    error_channel.log(err_str)
                    raise ValueError(err_str)

                # Weather model valid date of message 1
                grb_msg = grbs.message(1)
                weather_model_date = grb_msg.validDate
            else:
                #  Get the datetime of weather model in NetCDF format for RAiDER
                try:
                    # The HRES weather model file with ECMWF NetCDF format
                    if weather_model_type == 'HRES':
                        import xarray as xr
                        ds = xr.open_dataset(weather_model_file)
                        # Get the datetime of the weather model file
                        weather_model_date = ds.time.values.astype('datetime64[s]').astype(datetime)[0]
                    # The other weather model files with RAiDER NetCDF format
                    else:
                        with HDF5OptimizedReader(name=weather_model_file, mode='r', libver='latest', swmr=True) as f:
                                weather_model_date = datetime.strptime(f.attrs['datetime'].astype(str),
                                                                    '%Y_%m_%dT%H_%M_%S')
                except ValueError:
                    err_str = f'{weather_model_file} is not a netCDF format, and not supported by RAiDER package'
                    error_channel.log(err_str)
                    raise ValueError(err_str)

            diff = rslc_date - weather_model_date
            hours =  abs(diff.total_seconds()) / 3600.0

            # Check if it is more than 1 day (i.e. > 24 hours)
            if hours > 24.0:
                err_str = f"days difference between weather model ({weather_model_date})" + \
                          f" and RSLC ({rslc_date}) should be within one day"
                error_channel.log(err_str)
                raise ValueError(err_str)

        # Check the weather model
        weather_model_types = ['ERA5', 'ERAINT', 'HRES', 'NARR', 'MERRA',
                                'ECMWF', 'ERAI', 'GMAO', 'HRRR', 'NCMR']

        if weather_model_type not in weather_model_types:
            weather_model_types = ','.join(weather_model_types)
            err_str = f"unidentified weather model {weather_model_type}," + \
                        f'please try one of "{weather_model_types}"'
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Check the external tropo delay package
        if tropo_pkg not in ['pyaps', 'raider']:
            err_str = f"unidentified package {tropo_cfg['package']}," + \
                    " please use either pyaps or raider"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Further check the weather model that supported by pyAPS or RAiDER
        if (tropo_pkg  == 'pyaps') and \
                weather_model_type not in ['ERA5', 'ERAINT', 'HRES', 'NARR', 'MERRA']:
            err_str = f'weather model {weather_model_type} is not supported by pyAPS package'
            error_channel.log(err_str)
            raise ValueError(err_str)

        if (tropo_pkg  == 'raider') and \
                weather_model_type not in ['ERA5', 'HRES', 'MERRA', 'ECMWF',
                                           'ERAI', 'GMAO', 'HRRR', 'NCMR']:
            err_str = f'weather model {weather_model_type} is not supported by RAiDER package'
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Check the delay direction
        tropo_direction = tropo_cfg['delay_direction'].lower()
        if tropo_direction not in ['zenith',
                                   'line_of_sight_mapping',
                                   'line_of_sight_raytracing']:
            err_str = f"unidentified delay direction {tropo_cfg['delay_direction']}," + \
                        "please use one of 'zenith', 'line_of_sight_mapping'," + \
                        " 'line_of_sight_raytracing'"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # If the delay direction is 'line_of_sight_raytracing', the package must be 'raider'
        if tropo_direction == 'line_of_sight_raytracing' and \
                tropo_pkg != 'raider':
            err_str = "for line_of_sight_raytracing delay type, the package must be 'raider'"
            error_channel.log(err_str)
            raise ValueError(err_str)

class InsarTroposphereRunConfig(RunConfig):
    '''
    Troposhphere RunConfig
    '''
    def __init__(self, args):
        # InSAR submodules share a common "InSAR" schema
        super().__init__(args, 'insar')

        if self.args.run_config_path is not None:
            self.load_geocode_yaml_to_dict()
            self.geocode_common_arg_load()
            troposphere_delay_check(self.cfg)
