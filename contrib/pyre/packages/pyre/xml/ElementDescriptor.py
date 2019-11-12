# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


from .Descriptor import Descriptor


class ElementDescriptor(Descriptor):
    """
    Descriptor class that gathers all the metadata about a document tag that was provided by
    the user during the DTD declaration. It is used by DTD derived classes to decorate the
    Document instance and the tag handlers with the information needed by the Reader so it can
    process XML documents
    """


    # element meta data
    handler = None # the Node descendant that handles parsing events for this document element
    attributes = () # a list of the tag attribute descriptors that encode the document DTD


    # meta methods
    def __init__(self, *, tag, handler, root=False):
        super().__init__(name=tag)
        self.handler = handler
        self.root = root
        return


# end of file
