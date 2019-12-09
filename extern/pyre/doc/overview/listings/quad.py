#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import pyre
import gauss

# the application
class Quad(pyre.application):
    """
    A harness for {gauss} integrators
    """

    # public state
    samples = pyre.properties.int(default=10**5)
    integrator = pyre.facility(interface=gauss.integrator)

    # required interface
    @pyre.export
    def main(self, *args, **kwds):
        # pass the requested number of samples to the integrator
        self.integrator.samples = self.samples
        # get it to integrate
        integral = self.integrator.integrate()
        # print the answer
        print("integral = {}".format(integral))
        # return success
        return 0

    @pyre.export
    def main_mpi(self, *args, **kwds):
        # access the mpi package
        import mpi
        # find out how many tasks were launched
        size = mpi.world.size
        # find out my rank
        rank = mpi.world.rank
        # figure out how many samples to do and pass that on to my integrator
        self.integrator.samples = self.samples / size
        # integrate: average the estimates produced by each task
        integral = mpi.sum(self.integrator.integrate())/size
        # node 0: print the answer
        if rank == 0: print("integral = {}".format(integral))
        # all done
        return 0


# main
if __name__ == "__main__":
    # externals
    import sys
    # instantiate the application
    q = Quad(name='quad')
    # run it and return its exit code to the os
    sys.exit(q.run())


# end of file
