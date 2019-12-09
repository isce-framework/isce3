# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Node:
    """
    The base class for parsing event handlers
    """


    # public data
    tag = None
    elements = ()
    namespace = ""


    # interface
    def content(self, text, locator):
        """
        The handler for textual data within my body that is not associated with any of my children
        """
        # ignore such text by default
        # this handler may trigger multiple times as text is discovered surrounding the
        # processing of children nodes, or as whitespace is encountered
        return


    def newNode(self, *, name, attributes, locator):
        """
        The handler invoked when the opening tag for one of my children is encountered.

        The default implementation looks up the tag in my local dtd, retrieves the associated
        node factory, and invokes it to set up the context for handlingits content

        In typical use, there is no need to override this; but if you do, you should make sure
        to return a Node descendant that is properly set up to handle the contents of the named
        tag
        """
        # get the handler factory
        try:
            factory = self._pyre_nodeIndex[name]
        except KeyError as error:
           msg = "unknown tag {0!r}".format(name)
           raise self.DTDError(description=msg) from error

        # invoke it to get a new node for the parsing context
        try:
            node = factory(parent=self, attributes=attributes, locator=locator)
        except TypeError as error:
            msg = "could not instantiate handler for node {0!r}; extra attributes?".format(name)
            raise self.DTDError(description=msg) from error
        except KeyError as error:
            msg = "node {0!r}: unknown attribute {1!r}".format(name, error.args[0])
            raise self.DTDError(description=msg) from error

        # and return it
        return node


    def newQNode(self, *, name, namespace, attributes, locator):
        """
        The handler invoked when the opening tag for one of my namespace qualified children is
        encountered.

        See Node.newNode for details
        """
        # get the handler factory
        try:
            factory = self._pyre_nodeQIndex[(name, namespace)]
        except KeyError as error:
           msg = "unknown tag {0!r}".format(name)
           raise self.DTDError(description=msg) from error

        # invoke it to get a new node for the parsingcontext
        try:
            node = factory(parent=self, attributes=attributes, locator=locator)
        except TypeError as error:
            msg = "could not instantiate handler for node {0!r}; extra attributes?".format(name)
            raise self.DTDError(description=msg) from error
        except KeyError as error:
            msg = "node {0!r}: unknown attribute {1!r}".format(name, error.args[0])
            raise self.DTDError(description=msg) from error

        # and return it
        return node


    def notify(self, *, parent, locator):
        """
        The handler that is invoked when the parser encounters my closing tag
        """
        raise NotImplementedError(
            "class {.__name__!r} must override 'notify'".format(type(self)))


    # pull in the locator converter
    from . import newLocator
    newLocator = staticmethod(newLocator)


    # exceptions
    from .exceptions import DTDError


    # private data
    _pyre_nodeIndex = None
    _pyre_nodeQIndex = None


# end of file
