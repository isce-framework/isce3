#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Show me the 256 colors possible with the ANSI 256 escape sequences
"""

# framework
import pyre

# declaration
class ColorTable(pyre.application):
    """
    Build a 256 color table
    """

    @pyre.export
    def main(self, *args, **kwds):
        """
        The main entry point
        """
        # make a channel
        channel = self.debug
        # put the channel tag in its own line
        channel.line()
        # make the swatches
        for line in self.swatches():
            # show me
            channel.line(line)
        # flush
        channel.log()
        # all done
        return


    # implementation details
    def swatches(self):
        """
        Build the color swatches
        """
        # get my terminal
        term = self.pyre_executive.terminal
        # putting things back to normal
        normal = term.ansi['normal']
        # loop
        for r in range(6):
            # initialize a line
            row = []
            # the two inferior indices
            for g in range(6):
                for b in range(6):
                    # get the color sequence
                    color = term.rgb256(red=r, green=g, blue=b, foreground=False)
                    # add it to the pile
                    row += f"{color}  {normal}"
                # separate the swatches
                row += " "
            # a row is ready
            yield ''.join(row)

        # all done
        return


# main
if __name__ == "__main__":
    # build one
    app = ColorTable('colors')
    # runt it
    status = app.run()
    # and pass the result on
    raise SystemExit(status)


# end of file
