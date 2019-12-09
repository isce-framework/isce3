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

    python = pyre.externals.python()
    python.doc = "the python installation"


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
            # get my python
            python = self.python
        # if something went wrong
        except self.ConfigurationError as error:
            # show me
            self.error.log(str(error))
            # and bail
            return 0

        # show me
        info.line("python:")
        info.line("  package: {}".format(python))
        # if i have one
        if python:
            # how did i get this
            info.line("  locator: {}".format(python.pyre_where()))
            # version info
            info.line("  version: {.version}".format(python))
            info.line("  prefix: {.prefix}".format(python))
            # locations
            info.line("  tools:")
            info.line("    path: {}".format(python.join(python.bindir)))
            info.line("    interpreter: {.interpreter}".format(python))
            # compile line
            info.line("  compile:")
            info.line("    defines: {}".format(python.join(python.defines)))
            info.line("    headers: {}".format(python.join(python.incdir)))
            # link line
            info.line("  link:")
            info.line("    paths: {}".format(python.join(python.libdir)))
            info.line("    libraries: {}".format(python.join(python.libraries)))

            # get the configuration errors
            errors = python.pyre_configurationErrors
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
