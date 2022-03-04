import journal

import isce3
from nisar.workflows.runconfig import RunConfig

class GSLCRunConfig(RunConfig):
    def __init__(self, args):
        super().__init__(args, 'gslc')
        super().load_geocode_yaml_to_dict()
        super().geocode_common_arg_load()
        self.load()


    def load(self):
        '''
        Load GSLC specific parameters.
        '''
        if self.cfg['processing']['dem_margin'] is None:
            '''
            Default margin as the length of 50 pixels
            (max of X and Y pixel spacing).
            '''
            dem_file = self.cfg['dynamic_ancillary_file_group']['dem_file']
            dem_raster = isce3.io.Raster(dem_file)
            dem_margin = 50 * max([dem_raster.dx, dem_raster.dy])
            self.cfg['processing']['dem_margin'] = dem_margin
