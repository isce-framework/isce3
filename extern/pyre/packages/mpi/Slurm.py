# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import sys, subprocess
# the framework
import pyre
# my superclass
from .Launcher import Launcher


# declaration
class Slurm(Launcher, family='mpi.shells.slurm'):
    """
    Encapsulation of launching an MPI job using SLURM
    """


    # user configurable state
    sbatch = pyre.properties.path(default='sbatch')
    sbatch.doc = 'the path to the sbatch launcher'

    queue = pyre.properties.str()
    queue.doc = 'the name of the queue that will receive this job'

    submit = pyre.properties.bool(default=True)
    submit.doc = 'if {True} invoke sbatch; otherwise just save the SLURM script in a file'


    # spawning the application
    def spawn(self, application):
        """
        Generate a {SLURM} script and submit a job
        """
        # we have two things to build: the SLURM script, and the command line to {sbatch} to
        # submit the job to the queuing system

        # build the command line that we will include in the SLURM script
        argv = self.buildCommandLine()

        # meta-data
        # use the app name as the filename stem for the job name, stdout, and stderr
        stem = application.pyre_name
        # grab the name of the queue
        queue = self.queue
        # and the number of tasks
        tasks = self.tasks

        # here is the body of the script
        script = "\n".join([
            "#!/bin/bash",
            "",
            f"#SBATCH --job-name='{stem}'",
            f"#SBATCH --ntasks={tasks}",
            f"#SBATCH --output='{stem}.out'",
            f"#SBATCH --error='{stem}.err'",
            f"#SBATCH --partition='{queue}'",
            "",
            "# load the environment",
            "[ -r /etc/profile ] && source /etc/profile",
            "",
            "# launch the pyre application",
            " ".join(argv),
            "",
            "# end of file"
            ])

        # if we were asked not to invoke SLURM
        if not self.submit:
            # open a file named after the app
            with open(application.pyre_name+".slurm", "w") as record:
                # write the script
                record.write(script)
                # and return success
                return 0

        # grab the launcher
        sbatch = str(self.sbatch)
        # command line arguments
        options = {
            "args": [sbatch],
            "executable": sbatch,
            "stdin": subprocess.PIPE, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE,
            "universal_newlines": True,
            "shell": False
            }
        # invoke {sbatch}
        with subprocess.Popen(**options) as child:
            # send it the script
            response, errors = child.communicate(script)
            # if {sbatch} said anything
            if response: application.info.log(response)
            # if there was a problem
            if errors: application.error.log(errors)
            # wait for it to finish
            status = child.wait()
        # and return its status
        return status


# end of file
