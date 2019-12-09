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
    # get the modules
    import pyre.parsing
    import pyre.patterns

    # declare a scanner
    class Simple(pyre.parsing.scanner):
        """a simple scanner"""

        # tokens
        comment = pyre.parsing.token(r"#.*$")
        separator = pyre.parsing.token(r":")
        delimiter = pyre.parsing.token(r",")
        terminator = pyre.parsing.token(r";")
        identifier = pyre.parsing.token(r"[\w]+")

    # open the input stream
    uri = 'sample.inp'
    stream = open(uri)
    # create a source
    scanner = Simple()
    # and a sink
    sink = pyre.patterns.accumulator()


    # tokenize
    scanner.pyre_tokenize(uri=uri, stream=stream, client=sink)

    # check
    expected = (
        Simple.start,
        Simple.comment, Simple.whitespace,
        Simple.comment, Simple.whitespace,
        Simple.comment, Simple.whitespace,
        Simple.comment, Simple.whitespace,
        Simple.comment, Simple.whitespace,
        Simple.whitespace,
        Simple.whitespace, Simple.identifier, Simple.separator,
        Simple.whitespace, Simple.identifier, Simple.delimiter,
        Simple.whitespace, Simple.identifier, Simple.delimiter,
        Simple.whitespace, Simple.identifier, Simple.terminator, Simple.whitespace,
        Simple.whitespace,
        Simple.comment,
        Simple.whitespace,
        Simple.finish,
        )

    for token, klass in zip(sink.cache, expected):
        # print(token)
        assert isinstance(token, klass)

    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
