# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Indenter:
    """
    A mix-in class that keeps track of the indentation level
    """


    # public data
    leader = "" # the current contents to prepend to every line


    # interface
    def indent(self, increment=1):
        """
        Increase the indentation level by one
        """
        # increase the indentation level
        self._level += increment
        # and adjust the margin filler
        self.leader = self._indenter * self._level
        # all done
        return self


    def outdent(self, decrement=1):
        """
        Decrease the indentation level by one
        """
        # decrease the indentation level
        self._level -= decrement
        # and adjust the margin filler
        self.leader = self._indenter * self._level
        # all done
        return self


    def place(self, line):
        """
        Indent {line} and return it
        """
        # easy enough
        return self.leader + line


    # meta methods
    def __init__(self, indenter=None, level=0, **kwds):
        # chain up
        super().__init__(**kwds)
        # initialize my markers
        self._level = level
        self._indenter = self.INDENTER if indenter is None else indenter
        self.leader = self._indenter * level
        # all done
        return


    # constants
    INDENTER = " "*4


    # private data
    _level = 0
    _indenter = None


# end of file
