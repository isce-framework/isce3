import os

import journal

class Persistence():
    '''
    basic class that determines InSAR persistence
    '''
    # init InSAR steps in reverse chronological run order
    insar_steps = ['geocode', 'ionosphere', 'unwrap', 'filter_interferogram', 'crossmul',
                   'fine_resample', 'rubbersheet', 'offsets_product', 'dense_offsets',
                   'coarse_resample', 'geo2rdr', 'rdr2geo', 'h5_prep', 'bandpass_insar']

    def __init__(self, restart=False):
        # bool flag that determines if insar.run is called
        # prevents calling of insar.run if last run was successful and no restart
        self.run = restart

        # dict key: step name data: bool for whether or not to run step
        # assume all steps successfully ran so default each steps run flag to false
        self.run_steps = {}
        for i in self.insar_steps:
            self.run_steps[i] = restart

        if not restart:
            self.read_log()

        info_channel = journal.info("persistence.init")
        if self.run:
            info_channel.log("Possible steps to be run:")
            for step in self.insar_steps:
                info_channel.log(f"{step}: {self.run_steps[step]}")
        else:
            info_channel.log("No steps to be (re)run.")

    def read_log(self):
        '''
        determine state of last run to determine this runs steps
        '''
        # assume log file small enough to fit into memory for reverse read
        path_log = journal.debug.journal.device.log.name

        # check for empty log from error free runconfig and yamlparse execution
        if os.path.getsize(path_log) == 0:
            self.__init__(True)
        else:
            # read log in reverse chronological order
            for log_line in reversed(list(open(path_log, 'r'))):
                # check for end of successful run
                if 'successfully ran INSAR' in log_line:
                    break

                # success message found
                success_msg_found = False
                # check for message indicating successful run of step
                if 'uccessfully ran' in log_line:
                    # iterate thru reverse chronological steps
                    for insar_step in self.insar_steps:
                        # any step not found in line will be step to run
                        if insar_step not in log_line:
                            # set step name found to True
                            self.run_steps[insar_step] = True
                            success_msg_found = True
                        else:
                            # check if any steps need to be run
                            if any(self.run_steps.values()):
                                self.run = True

                            # all previous steps successfully run and stop
                            break

            # check if any steps need to be run or success msg not found
            if not success_msg_found:
                self.__init__(True)
