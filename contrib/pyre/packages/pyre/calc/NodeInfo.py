# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# declaration
class NodeInfo:
    """
    The base class for nodal metadata maintained by symbol tables
    """


    # public data
    key = None # the hashed version of the symbol name
    name = None # the string version of the symbol name
    split = None # the symbol name split on the table separator


    # interface
    @staticmethod
    def fillNodeId(model, key=None, name=None, split=None):
        """
        Given one of the three representations of the key of a node in {model}, reconstruct all of
        them so clients can choose whichever representation fits their needs
        """
        # if I know the name but not the split version
        if name and not split:
            # set the split
            split = tuple(model.split(name))
        # otherwise, if I know the split but not the name
        elif split and not name:
            # get the name
            name = model.join(*split)

        # if I don't know the key but I know the split
        if split and not key:
            # look up the key
            key = model._hash.hash(items=split)

        # done my best: if i know the key
        if key:
            # return the info
            return name, split, key

        # otherwise, get the journal
        import journal
        # and complain; this is a bug
        raise journal.firewall('pyre.calc').log('insufficient nodal metadata')


    # meta-methods
    def __init__(self, key=None, name=None, split=None, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the key information
        self.key = key
        self.split = split
        self.name = name
        # all done
        return


# end of file
