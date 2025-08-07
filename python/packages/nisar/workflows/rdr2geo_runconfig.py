import journal

from nisar.workflows.runconfig import RunConfig


class Rdr2geoRunConfig(RunConfig):
    def __init__(self, args):
        # all insar submodules share a common `insar` schema
        super().__init__(args, 'insar')

        if self.args.run_config_path is not None:
            self.load_geocode_yaml_to_dict()
            self.geocode_common_arg_load()
            self.yaml_check()

    def yaml_check(self):
        error_channel = journal.error('Rdr2GeoRunConfig.yaml_check')
        layers = ['x', 'y', 'z', 'incidence', 'heading', 'local_incidence',
                  'local_psi', 'simulated_amplitude', 'layover_shadow']

        # get rdr2geo config dict from processing dict for brevity
        rdr2geo_cfg = self.cfg['processing']['rdr2geo']

        # list comprehend rasters to be written from layers dict
        write_any_layer = any([rdr2geo_cfg[f'write_{layer}'] for layer in layers])
        if not write_any_layer:
            err_str = "All topo layers disabled"
            error_channel.log(err_str)
            raise ValueError(err_str)
