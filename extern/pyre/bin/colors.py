#!/usr/bin/env python3
# -*- Python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# support
import pyre


# the app
class Colors(pyre.application):
    """
    A generator of colorized directory listings that is repository aware
    """


    # user configurable state
    palette = pyre.properties.str(default="x11")
    palette.doc = "the set of colors to render"

    # protocol obligations
    @pyre.export
    def main(self, *args, **kwds):
        """
        The main entry point
        """
        # grab my terminal
        terminal = self.executive.terminal
        # get the reset code
        normal = terminal.ansi["normal"]
        # get the palette
        palette = getattr(terminal, self.palette)
        # go through the color names
        for name in palette.keys():
            # print the name in its color
            print(f"{palette[name]}{name}{normal}")
        #
        # all done
        return 0



# bootstrap
if __name__ == "__main__":
    # instantiate
    app = Colors(name="colors")
    # invoke
    status = app.run()
    # share
    raise SystemExit(status)


# end of file
