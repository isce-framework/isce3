# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# declaration
class Script:
    """
    A locator that records information relevant to python scripts. This information is
    typically extracted from stack traces so it contains whatever can be harvested by
    introspection
    """


    # meta methods
    def __init__(self, source, line=None, function=None):
        # save my info
        self.source = source
        self.line = line
        self.function = function
        # all done
        return


    def __str__(self):
        text = [
            "file={!r}".format(str(self.source))
            ]
        if self.line:
            text.append("line={.line!r}".format(self))
        if self.function:
            text.append("function={.function!r}".format(self))

        return ", ".join(text)


    # implementation details
    __slots__ = "source", "line", "function"


# end of file
