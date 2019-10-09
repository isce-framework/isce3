# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# get the symbolic names for the ASCII codes
from .ASCII import ASCII


# declaration
class CSI:
    """
    A generator of ANSI control strings
    """


    # reset
    @staticmethod
    def reset():
        """
        Reset all output attributes
        """
        # build the sequence and return it
        return f"{ASCII.ESC}[0m"


    # the color commands
    @staticmethod
    def csi3(bright=False, code=None):
        """
        Build an ANSI color sequence
        """
        # build the sequence
        seq = [
            f"{ASCII.ESC}[",
            "1" if bright else "0",
            f";{code}" if code is not None else "",
            "m"
        ]
        # assemble it and return it
        return "".join(seq)


    @staticmethod
    def csi8(red=0, green=0, blue=0, foreground=True):
        """
        Build an ANSI color sequence from the 256 color set, where each color can take a value in
        the interval [0, 5]
        """
        # build the sequence
        seq = [
            f"{ASCII.ESC}[",
            "38" if foreground else "48",
            ";5;",
            str(16 + int(f"{red}{green}{blue}", 6)),
            "m"
        ]
        # assemble it and return it
        return "".join(seq)


    @staticmethod
    def csi8_gray(gary=0, foreground=True):
        """
        Build an ANSI color sequence from the 8 bit color set that corresponds to a gray level in
        the range [0, 23]
        """
        # build the sequence
        seq = [
            f"{ASCII.ESC}[",
            "38" if foreground else "48",
            ";5;",
            str(232 + gray),
            "m"
        ]
        # assemble it and return it
        return "".join(seq)


    @staticmethod
    def csi24(red=0, green=0, blue=0, foreground=True):
        """
        Build an ANSI color sequence from the 24bit color set, where each color can take a value in
        the interval [0, 255]
        """
        # build the sequence
        seq = [
            f"{ASCII.ESC}[",
            "38" if foreground else "48",
            f";2;{red};{green};{blue}",
            "m"
        ]
        # assemble it and return it
        return "".join(seq)


    # graphics rendition commands
    @staticmethod
    def blink(state=True):
        """
        Turn blink on or off
        """
        # build the sequence
        seq = [
            f"{ASCII.ESC}["
            "5" if state else "25"
            "m"
        ]
        # assemble it and return it
        return "".join(seq)


# end of file
