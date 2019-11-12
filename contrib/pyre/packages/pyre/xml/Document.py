# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


from .DTD import DTD
from .Node import Node


class Document(Node, metaclass=DTD):
    """
    Base class for the custom processing of XML parsing events. You must derive from this class
    and declare the element descriptors that correspond to the tags in your document.
    Instances of your derived class will form the interface between the XML parser and the
    application specific data structure being decorated with the contents of the XML stream.

    This object is the anchor for the handler of the top element handler that is reposinsible
    for the root of the XML document.

    The DTD metaclass scans through the class record, identifies the element declarations and
    builds the DTF for the document
    """


    # public data
    # inherited
    tag = "document" # mark this as the top-level document object
    root = None # descendants must provide the name of the root element here
    elements = () # replace with a container with the tag for the top level document element
    # new
    dtd = None
    dom = None # the client data structure i will be decorating


    # interface
    def initialize(self, locator):
        """
        Handler for the event generated when parsing the XML document has just begun
        """
        return


    def finalize(self, locator):
        """
        Handler for the event generated when parsing of the XML document is finished
        """
        return


    # private data
    # inherited from Node
    _pyre_nodeIndex = None


# end of file
