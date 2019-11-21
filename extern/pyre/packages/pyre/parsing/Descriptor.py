# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Descriptor:
    """
    Place holders for the token specifications

    Descriptors are harvested by {Lexer}, the metaclass of {Scanner} subclasses, and converted
    into subclasses of {Token}
    """

    # public data
    head = '' # a pattern for text required for a match that is not part of the lexeme
    tail = '' # a pattern for text required for a match that is not part of the lexeme
    pattern = '' # the regular expression that extracts the lexeme


    # meta methods
    def __init__(self, pattern=None, head='', tail='', **kwds):
        # chain up
        super().__init__(**kwds)
        # save my parts
        self.head = head
        self.tail = tail
        self.pattern = pattern
        # all done
        return


    def __str__(self):
        """
        Build a representation of the descriptor, mostly for debugging purposes
        """
        return "{{head: '{}', pattern: '{}', tail: '{}'}}".format(
            self.head, self.pattern, self.tail)


# end of file
