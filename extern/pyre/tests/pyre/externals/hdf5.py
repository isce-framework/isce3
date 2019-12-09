#!/usr/bin/env python.pyre
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# the framework
import pyre


# the app
class configure(pyre.application):
    """
    A sample configuration utility
    """

    hdf5 = pyre.externals.hdf5()
    hdf5.doc = "the HDF5 installation"


    @pyre.export
    def main(self, *args, **kwds):
        """
        The main entry point
        """
        # get my channel
        info = self.info
        # show me
        info.line("{.pyre_name}:".format(self))
        info.line("  host: {.pyre_host}".format(self))
        info.line("  package manager: {.pyre_host.packager}".format(self))
        # flush
        info.log()

        # attempt to
        try:
            # get my hdf5
            hdf5 = self.hdf5
        # if something went wrong
        except self.ConfigurationError as error:
            # show me
            self.error.log(str(error))
            # and bail
            return 0

        # show me
        info.line("hdf5:")
        info.line("  package: {}".format(hdf5))
        # if i have one
        if hdf5:
            # how did i get this
            info.line("  locator: {}".format(hdf5.pyre_where()))
            # version info
            info.line("  version: {.version}".format(hdf5))
            info.line("  prefix: {.prefix}".format(hdf5))
            # compile line
            info.line("  compile:")
            info.line("    defines: {}".format(hdf5.join(hdf5.defines)))
            info.line("    headers: {}".format(hdf5.join(hdf5.incdir)))
            # link line
            info.line("  link:")
            info.line("    paths: {}".format(hdf5.join(hdf5.libdir)))
            info.line("    libraries: {}".format(hdf5.join(hdf5.libraries)))

            # get the configuration errors
            errors = hdf5.pyre_configurationErrors
            # if there were any
            if errors:
                # tell me
                info.line("  configuration errors that were auto-corrected:")
                # and show me
                for index, error in enumerate(errors):
                    info.line("      {}: {}".format(index+1, error))
        # flush
        info.log()

        # all done
        return 0


# main
if __name__ == "__main__":
    # get the journal
    import journal
    # activate the debug channel
    # journal.debug("app").activate()
    # make one
    app = configure(name='configure')
    # drive
    status = app.run()
    # all done
    raise SystemExit(status)


# end of file
