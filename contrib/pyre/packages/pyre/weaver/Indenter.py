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
        self._level += increment
        self.leader = self._indenter * self._level
        return self


    def outdent(self, decrement=1):
        """
        Decrease the indentation level by one
        """
        self._level -= decrement
        self.leader = self._indenter * self._level
        return self


    def place(self, line):
        return self.leader + line


    # meta methods
    def __init__(self, indenter=None, **kwds):
        # chain up
        super().__init__(**kwds)
        # initialize my markers
        self.leader = ""
        self._level = 0
        self._indenter = self.INDENTER if indenter is None else indenter
        # all done
        return


    # constants
    INDENTER = " "*4


    # private data
    _level = 0
    _indenter = None


# end of file
