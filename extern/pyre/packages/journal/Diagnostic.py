# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# packages
import pyre
import traceback


# declaration
class Diagnostic:
    """
    Encapsulation of the message recording behavior of channels
    """


    # per-instance public data
    meta = None
    text = None
    locator = None


    # interface
    def line(self, message=''):
        """
        Add {message} to the diagnostic text
        """
        # check whether i am an active diagnostic
        if self.active:
            # add {message} to my text
            self.text.append(message)
        # and return
        return self


    def log(self, message=None, stackdepth=0):
        """
        Add the optional {message} to my text and make a journal entry
        """
        # bail if I am not active
        if not self.active: return self
        # if {message} is non-empty, add it to the pile
        if message is not None: self.text.append(message)

        # use the stack
        trace = traceback.extract_stack(limit=self.stackdepth+stackdepth)
        # to infer some more meta data
        filename, line, function, source = trace[0]
        # decorate
        meta = self.meta
        meta["filename"] = filename
        meta["line"] = line
        meta["function"] = function
        meta["source"] = source

        # build my locator
        self.locator = pyre.tracking.script(source=filename, line=line, function=function)

        # record
        self.device.record(page=self.text, metadata=meta)

        # clean up
        self.text = []

        # and return
        return self


    # meta methods
    def __init__(self, name, **kwds):
        # chain to the ancestors
        super().__init__(name=name, **kwds)

        # initialize the list of message lines
        self.text = []
        # my locator
        self.locator = None
        # prime the meta data
        self.meta = {
            "channel": name,
            "severity": self.severity
            }

        # all done
        return


    # implementation details
    # the stack depth of my clients
    stackdepth = pyre.computeCallerStackDepth()


# end of file
