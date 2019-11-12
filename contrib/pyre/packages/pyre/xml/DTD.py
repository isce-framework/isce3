# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


from ..patterns.AttributeClassifier import AttributeClassifier


class DTD(AttributeClassifier):
    """
    Metaclass that scans the class record of a Document descendant for element descriptors and
    builds the necessary machinery for parsing the XML document
    """


    # types
    from .Descriptor import Descriptor


    # meta methods
    def __new__(cls, name, bases, attributes, **kwds):
        """
        Build the document class record
        """
        # storage for the document nodes
        nodes = []
        # harvest them
        for name, node in cls.pyre_harvest(attributes, cls.Descriptor):
            # record the name name
            node.name = name
            # and add it to the pile
            nodes.append(node)

        # record them as the DTD
        attributes["dtd"] = dtd = tuple(nodes)
        # chain up to build the node
        node = super().__new__(cls, name, bases, attributes, **kwds)

        # namespaces introduce a bit of complexity. unless it turns out to be inconsistent with
        # the rules, here is the strategy: if a nested element is in the same namespace as its
        # direct parent, it goes into the _pyre_nodeIndex. otherwise it goes into the
        # _pyre_nodeQIndex of namespace qualified tags

        # in the Reader, {start|end}Element look up tags directly in the _pyre_nodeIndex, which
        # is the only possible implementation since there is no additional information
        # available beyond the tag name. this is equivalent to assuming that the nested tag
        # belongs to the same namespace. the namespace qualified hooks {start|end}ElementNS
        # need a slightly modified approach: if the namespace given matches the namespace of
        # the parent tag, look in _pyre_nodeIndex; if not look it up in _pyre_nodeQIndex

        # build a (element name -> handler) map
        index = { element.name: element for element in dtd }

        # now, build the dtd for each handler
        for element in dtd:
            # get the class that handles this element
            handler = element.handler
            # initialize the class attributes
            handler.tag = element.name
            handler._pyre_nodeIndex = {}
            handler._pyre_nodeQIndex = {}
            # if this is the root element
            if element.root:
                # record it in the document class
                node.root = element.name
            # build the nested element indices
            for tag in handler.elements:
                # get the nestling handler
                nestling = index[tag].handler
                # figure out which into which index this nestling should be placed
                if handler.namespace == nestling.namespace:
                    handler._pyre_nodeIndex[tag] = nestling
                else:
                    handler._pyre_nodeQIndex[(nestling.namespace, tag)] = nestling

        # and now adjust the actual document class
        if node.root is not None:
            root = index[node.root].handler
            node.namespace = root.namespace
            node._pyre_nodeIndex = { node.root: root }

        # return the node to the caller
        return node


# end of file
