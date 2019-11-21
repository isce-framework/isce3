# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import os
# access the framework
import pyre
# get my protocol
from .Terminal import Terminal as terminal
# and the escape sequence helper
from .CSI import CSI


# declaration
class ANSI(pyre.component, family='pyre.terminals.ansi', implements=terminal):
    """
    A terminal that provides color capabilities using ANSI control sequences
    """


    # public data
    @property
    def width(self):
        """
        Compute the width of the terminal
        """
        # attempt to
        try:
            # ask python
            return os.get_terminal_size().columns
        # if something went wrong
        except OSError:
            # absorb
            pass
        # don't know
        return 0


    # interface
    def rgb(self, rgb, foreground=True):
        """
        Mix the 6 digit hex string into an ANSI 24-bit color
        """
        # unpack and convert
        red, green, blue = (int(rgb[2*pos:2*(pos+1)], 16) for pos in range(3))
        # build the control sequence
        return CSI.csi24(red=red, green=green, blue=blue, foreground=foreground)


    def rgb256(self, red=0, green=0, blue=0, foreground=True):
        """
        Mix the three digit (r,g,b) base 6 string into an ANSI 256 color
        """
        # build the control sequence
        return CSI.csi8(red=red, green=green, blue=blue, foreground=foreground)


    # meta-methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # figure out the emulation
        self.emulation = os.environ.get('TERM', 'unknown').lower()
        # all done
        return


    # the ansi color names with their standard implementations
    ansi = {
        "": "", # no color given
        "none": "", # no color
        "normal": CSI.csi3(code="0"), # reset back to whatever is the default for the terminal

        # regular colors
        "black": CSI.csi3(code="30"),
        "red": CSI.csi3(code="31"),
        "green": CSI.csi3(code="32"),
        "brown": CSI.csi3(code="33"),
        "blue": CSI.csi3(code="34"),
        "purple": CSI.csi3(code="35"),
        "cyan": CSI.csi3(code="36"),
        "light-gray": CSI.csi3(code="37"),

        # bright colors
        "dark-gray": CSI.csi3(bright=True, code="30"),
        "light-red": CSI.csi3(bright=True, code="31"),
        "light-green": CSI.csi3(bright=True, code="32"),
        "yellow": CSI.csi3(bright=True, code="33"),
        "light-blue": CSI.csi3(bright=True, code="34"),
        "light-purple": CSI.csi3(bright=True, code="35"),
        "light-cyan": CSI.csi3(bright=True, code="36"),
        "white": CSI.csi3(bright=True, code="37"),
        }


    # the X11 named colors
    x11 = {
        "burlywood": CSI.csi24(red=0xde, green=0xb8, blue=0x87),
        "dark_goldenrod": CSI.csi24(red=0xb8, green=0x86, blue=0x0b),
        "dark_khaki": CSI.csi24(red=0xbd, green=0xb7, blue=0x6b),
        "dark_orange": CSI.csi24(red=0xff, green=0x8c, blue=0x00),
        "dark_sea_green": CSI.csi24(red=0x8f, green=0xbc, blue=0x8f),
        "firebrick": CSI.csi24(red=0xb2, green=0x22, blue=0x22),
        "hot_pink": CSI.csi24(red=0xff, green=0x69, blue=0xb4),
        "indian_red": CSI.csi24(red=0xcd, green=0x5c, blue=0x5c),
        "lavender": CSI.csi24(red=0xc0, green=0xb0, blue=0xe0),
        "light_green": CSI.csi24(red=0x90, green=0xee, blue=0x90),
        "light_steel_blue": CSI.csi24(red=0xb0, green=0xc4, blue=0xde),
        "light_slate_gray": CSI.csi24(red=0x77, green=0x88, blue=0x99),
        "lime_green": CSI.csi24(red=0x32, green=0xcd, blue=0x32),
        "navajo_white": CSI.csi24(red=0xff, green=0xde, blue=0xad),
        "olive_drab": CSI.csi24(red=0x6b, green=0x8e, blue=0x23),
        "peach_puff": CSI.csi24(red=0xff, green=0xda, blue=0xb9),
        "sage": CSI.csi24(red=176, green=208, blue=176),
        "steel_blue": CSI.csi24(red=0x46, green=0x82, blue=0xb4),
    }


    # grays
    gray = {
        "gray10": CSI.csi24(red=0x19, green=0x19, blue=0x19),
        "gray30": CSI.csi24(red=0x4c, green=0x4c, blue=0x4c),
        "gray41": CSI.csi24(red=0x69, green=0x69, blue=0x69),
        "gray50": CSI.csi24(red=0x80, green=0x80, blue=0x80),
        "gray66": CSI.csi24(red=0xa9, green=0xa9, blue=0xa9),
        "gray75": CSI.csi24(red=0xbe, green=0xbe, blue=0xbe),
    }


    # other
    misc = {
        "amber": CSI.csi24(red=0xff, green=0xbf, blue=0x00),
    }


# end of file
