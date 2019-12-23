# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# the id generator
import uuid


# declaration
class NameGenerator:
    """
    A generator of globally unique names
    """


    # public data
    # the legal symbols in value order; a scramble of digits and letters
    alphabet = "V7MWCTZESF6DQU5IX0JOBG3YAPKN248HLR91"


    # interface
    def uid(self):
        """
        Return the next name in the sequence
        """
        # make an id
        uid = uuid.uuid1().int
        # convert it into a string
        name = "".join(self._encode(uid))
        # pad it to a multiple of 4 by pulling letters in alphabet order
        name += self.alphabet[:4-len(name)%4]
        # make it more legible
        name = "-".join(name[q:q+4] for q in range(0, len(name), 4))
        # and return it
        return name


    # meta-methods
    def __init__(self, alphabet=alphabet, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the alphabet
        self.alphabet = alphabet
        # compute the implied base
        self.base = len(alphabet)
        # all done
        return


    # implementation details
    def _encode(self, uid):
        """
        Convert {uid} into a number using our {alphabet} for symbols
        """
        # grab the alphabet
        alphabet = self.alphabet
        # and the implied base
        base = self.base

        # per Euclid
        while 1:
            # split
            uid, remainder = divmod(uid, base)
            # hash and return a character
            yield alphabet[remainder]
            # if we have reduced the {uid} completely
            if uid == 0:
                # nothing further to do
                break

        # all done
        return


# end of file
