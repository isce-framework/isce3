#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Launch an mpi application
"""


def test():
    # access the framework
    import pyre

    # declare a trivial application
    class application(pyre.application, family='mpi.application'):
        """a sample application"""

        @pyre.export
        def main(self, **kwds):
            # access the package
            import mpi
            # get the world communicator
            world = mpi.world
            # print("Hello from {0.rank} of {0.size}".format(world))
            # check
            assert world.size == self.shell.tasks
            assert world.rank in range(world.size)
            # all done
            return 0

    # instantiate it
    app = application(name='slurm')
    # attempt to
    try:
        # run it
        app.run()
    # if this fails because we don't have slurm
    except FileNotFoundError as error:
        # the name of the slurm submission script
        sbatch = 'sbatch'
        # make sure that we are just missing {sbatch}
        assert error.errno == 2
        assert error.filename == f"{sbatch}"

    # return the app
    return app


# main
if __name__ == "__main__":
    test()


# end of file
