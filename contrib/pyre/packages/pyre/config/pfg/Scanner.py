# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import pyre.parsing


# the scanner
class Scanner(pyre.parsing.sws):
    """
    Converts an input source into a stream of tokens.
    """


    # types
    pyre_token = pyre.parsing.sws.pyre_token

    # my tokens, in addition to the three inherited from {pyre.parsing.scanner}
    section = pyre_token(pattern=r'\w[-.\w]*(\s*#\s*\w[-.\w]*)?', tail='\s*:')
    assignment = pyre_token(pattern=r'[^=]+', tail='\s*=')
    comment = pyre_token(head=r'(?<!\\);', pattern='.*', tail='$')


    # interface
    def pyre_tokenize(self, uri, stream, client):
        """
        Convert the input {stream} into tokens that are not whitespace
        """
        # adjust the client
        filtered = self.pyre_ignoreWhitespace(client)
        # and process the token stream
        return super().pyre_tokenize(uri=uri, stream=stream, client=filtered)


# end of file
