#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Perform terminal output using the support for 24 bit color
"""

# framework
import pyre

# declaration
class ColorTable(pyre.application):
    """
    Exercise the 24-bit color capabilities
    """

    # public state
    color = pyre.properties.str(default='c0c0c0')
    color.doc = 'a 6 digit hex string with the desired color value'


    # behavior
    @pyre.export
    def main(self, *args, **kwds):
        """
        The main entry point
        """
        # get my terminal
        term = self.pyre_executive.terminal
        # render the user selected color
        color = term.rgb(rgb=self.color, foreground=False)
        # putting things back to normal
        normal = term.ansi['normal']
        # show me
        self.debug.log(f"{color}Hello!{normal}")
        # all done
        return 0


# main
if __name__ == "__main__":
    # build one
    app = ColorTable('colors')
    # runt it
    status = app.run()
    # and pass the result on
    raise SystemExit(status)


# end of file
