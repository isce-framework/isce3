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
    import pyre.parsing

    COMMENT = r"#"
    SEPARATOR = r":"

    class Simple(pyre.parsing.scanner):
        """a simple scanner"""
        comment = pyre.parsing.token(COMMENT)
        separator = pyre.parsing.token(SEPARATOR)


    # access the token base class
    from pyre.parsing.Token import Token

    # check that the token descriptors have been turned into subclasses of Token
    assert issubclass(Simple.comment, Token)
    assert issubclass(Simple.separator, Token)
    # check that the tokenizer was built correctly
    assert Simple.pyre_tokenizer.pattern == '|'.join([
        "(?P<comment>#)",
        "(?P<separator>:)",
        "(?P<whitespace>\s+)",
        ])

    # and return the class record
    return Simple


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
