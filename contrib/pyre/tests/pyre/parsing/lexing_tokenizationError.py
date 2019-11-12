#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Build and test a simple tokenizer
"""

def test():
    # get the necessary packages
    import pyre.parsing
    import pyre.patterns

    # scanner
    class Simple(pyre.parsing.scanner):
        """a simple scanner"""

        # tokens
        comment = pyre.parsing.token(r"#.*$")
        separator = pyre.parsing.token(r":")
        delimiter = pyre.parsing.token(r",")
        terminator = pyre.parsing.token(r";")
        identifier = pyre.parsing.token(r"[_\w]+")


    class Sink(pyre.patterns.accumulator):
        """a sink that catches errors"""

        def throw(self, errorTp, error=None, traceback=None):
            # place the error on myp pile
            self.errors.append(error)
            # if this were a TokenizationError:
            if errorTp is Simple.TokenizationError:
                # ignore it
                return
            # pass everything else on
            super().throw(errorTp, error, traceback)


        def __init__(self, **kwds):
            # the pile of errors
            self.errors =  []
            # chain up
            super().__init__(**kwds)
            # all done
            return


    # the source
    filename = "sample-bad.inp"
    # open the input stream
    stream = open(filename)
    # create a source
    scanner = Simple()
    # and some sinks
    sink = Sink()
    printer = pyre.patterns.printer()
    tee = pyre.patterns.tee([sink, printer])

    # tokenize; this completes successfully, even though there are errors in the input
    scanner.pyre_tokenize(uri=filename, stream=stream, client=sink)

    # we expect two errors
    assert len(sink.errors) == 2

    # the first
    error = sink.errors[0]
    # is at line 7, column 31
    assert error.locator.source == filename
    assert error.locator.line == 7
    assert error.locator.column == 31

    # the second
    error = sink.errors[1]
    # is at line 8, column 24
    assert error.locator.source == filename
    assert error.locator.line == 8
    assert error.locator.column == 24

    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
