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

    gcc = pyre.externals.gcc()
    gcc.doc = "the GCC installation"


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
            # get my gcc
            gcc = self.gcc
        # if something goes wrong
        except self.ConfigurationError as error:
            # show me
            self.error.log(str(error))
            # and bail
            return 0

        # show me
        info.line("gcc:")
        info.line("  package: {}".format(gcc))
        # if i have one
        if gcc:
            # how did i get this
            info.line("  locator: {}".format(gcc.pyre_where()))
            # version info
            info.line("  version: {.version}".format(gcc))
            info.line("  prefix: {.prefix}".format(gcc))
            # locations
            info.line("  tool:")
            info.line("    path: {}".format(gcc.join(gcc.bindir)))
            info.line("    wrapper: {.wrapper}".format(gcc))

            # get the configuration errors
            errors = gcc.pyre_configurationErrors
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
