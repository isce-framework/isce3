# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# declaration
class FileRegion:
    """
    A locator that records information about a region of a file
    """


    # meta methods
    def __init__(self, start, end):
        self.start = start
        self.end = end
        return


    def __str__(self):
        # start of the region
        start = []
        if self.start.line:
            start.append("line={.line!r}".format(self.start))
        if self.start.column:
            start.append("column={.column!r}".format(self.start))
        start = ", ".join(start)

        # end of the region
        end = []
        if self.end.line:
            end.append("line={.line!r}".format(self.end))
        if self.end.column:
            end.append("column={.column!r}".format(self.end))
        end = ", ".join(end)

        text = "file={!r}, from ({}) to ({})".format(str(self.start.source), start, end)

        return text


    # implementation details
    __slots__ = "start", "end"


# end of file
