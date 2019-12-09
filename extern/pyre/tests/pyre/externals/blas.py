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

    blas = pyre.externals.blas()
    blas.doc = "the BLAS installation"


    @pyre.export
    def main(self, *args, **kwds):
        """
        The main entry point
        """
        # get my channel
        info = self.info
        # show me
        info.line("{.pyre_name}:".format(self))
        info.line("  host: {.pyre_host.nickname!r}".format(self))
        info.line("  package manager: {.pyre_host.packager.name!r}".format(self))
        # flush
        info.log()

        # attempt to
        try:
            # get my blas
            blas = self.blas
        # if something went wrong
        except self.ConfigurationError as error:
            # show me
            self.error.log(str(error))
            # and bail
            return 0

        # show me
        info.line("blas:")
        info.line("  package: {}".format(blas))
        # if i have one
        if blas:
            # how did i get this
            info.line("  locator: {}".format(blas.pyre_where()))
            # version info
            info.line("  version: {.version}".format(blas))
            info.line("  prefix: {.prefix}".format(blas))
            # compile line
            info.line("  compile:")
            info.line("    defines: {}".format(blas.join(blas.defines)))
            info.line("    headers: {}".format(blas.join(blas.incdir)))
            # link line
            info.line("  link:")
            info.line("    paths: {}".format(blas.join(blas.libdir)))
            info.line("    libraries: {}".format(blas.join(blas.libraries)))

            # get the configuration errors
            errors = blas.pyre_configurationErrors
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
