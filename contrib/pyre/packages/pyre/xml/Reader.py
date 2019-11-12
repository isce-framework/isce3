# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


import xml.sax
from . import newLocator


class Reader(xml.sax.ContentHandler):
    """
    An event driver reader for XML documents
    """


    # types
    # exceptions
    from .exceptions import DTDError, ParsingError, ProcessingError, UnsupportedFeatureError


    # public data
    ignoreWhitespace = False


    # features
    from xml.sax.handler import (
        feature_namespaces, feature_namespace_prefixes, feature_string_interning,
        feature_validation, feature_external_ges, feature_external_pes
        )


    # interface
    def read(self, *, stream, document, features=(), saxparser=None):
        """
        Build a representation of the information in {stream}

        parameters:
            {stream}: a URI or file-like object
            {document}: an instance of the Document data structure to be decorated with the
                        contents of {stream}
            {saxparser}: the SAX style parser to use; defaults to xml.sax.make_parser()
            {features}: the optional parsing features to enable; expected to be a tuple of
                        (feature, value) pairs; for more details, see the built in package
                        xml.sax or your parser's documentation

        The {Reader} attempts to normalize the exceptions generated while parsing the XML
        document by converting them to one of the exception classes in this package. This
        mechanism fails if you supply your own parser, so you must be ready to catch any
        exceptions it may generate.
        """

        # attach the document
        self._document = document
        # create a parser
        parser = saxparser or xml.sax.make_parser()

        # apply the optional features
        unsupported = []
        for feature, value in features:
            try:
                parser.setFeature(feature, value)
            except xml.sax.SAXNotSupportedException:
                unsupported.append(feature)
        # raise an exception if any requests could not be satisfied
        if unsupported:
            raise self.UnsupportedFeatureError(self, document, unsupported)

        # parse
        parser.setContentHandler(self)
        try:
            parser.parse(stream)
        except xml.sax.SAXParseException as error:
            # something bad happened; normalize the exception
            raise self.ProcessingError(
                parser=self, document=document, description=error.getMessage(),
                saxlocator=self._locator) from error
        except self.ParsingError as error:
            error.parser = self
            error.document = document
            raise

        # clean up
        parser.setContentHandler(None)
        # and return the decorated data structure
        return self._document.dom


    # content handling: these methods are called by the underlying parser
    def startDocument(self):
        """
        Handler for the beginning of the document
        """
        # print(
            # "line {0}, col {1}: start document".format(
                # self._locator.getLineNumber(), self._locator.getColumnNumber()))

        # initialize the parsing variables
        self._nodeStack = []
        self._currentNode = self._document

        # notify the document that parsing has begun
        self._document.initialize(locator=self._locator)

        return


    def startElement(self, name, attributes):
        """
        Handler for the beginning of an element
        """
        # print(
            # "line {0}, col {1}: start element {2!r}".format(
                # self._locator.getLineNumber(), self._locator.getColumnNumber(), name))

        # get the current node to build me a rep for the requested child node
        try:
            node = self._currentNode.newNode(
                name=name, attributes=attributes, locator=self._locator)
        # catch DTDErrors and decorate them
        except self.DTDError as error:
            error.parser = self
            error.document = self._document
            error.locator = newLocator(self._locator)
            raise

        # if the current node created a valid child
        if node is not None:
            # push the current node on the stack
            self._nodeStack.append(self._currentNode)
            # and make the new node the focus of attention
            self._currentNode = node

        return


    def startElementNS(self, name, qname, attributes):
        """
        Handler for the beginning of an element when feature_namespaces is turned on and the
        element encountered has a namespace qualification, either explicitly given with the
        tag, or because the document specifies a default namespace
        """

        # NYI:
        #     qnames seem to alway be None, so I am ignoring them
        #     will address later, after further explorations

        # unpack the qualified name
        namespace, tag = name
        # normalize the attributes
        # NYI:
        #     currently, we only supportthe case where the attribute names belong to the same
        #     namespace as the element itself
        # print(" *** attributes:", attributes)
        # print(" ***      names:", attributes.getNames())
        # print(" ***     qnames:", attributes.getQNames())
        # print(" ***   contents:", attributes.items())
        normalized = {}
        for ((namespace, name), value) in attributes.items():
            normalized[name] = value

        # use the correct factory to build a new Node instance
        try:
            # print(" startElementNS:")
            # print("            current tag:", self._currentNode.tag)
            # print("      current namespace:", self._currentNode.namespace)
            # print("          requested tag:", tag)
            # print("    requested namespace:", namespace)

            # namespace is set to None if the element has namespace decoration, in the same
            # namespace as its parent and there is no default namespace declared by the
            # document
            if namespace is None or namespace == self._currentNode.namespace:
                node = self._currentNode.newNode(
                    name=tag, attributes=normalized, locator=self._locator)
            else:
                node = self._currentNode.newQNode(
                    name=name, namespace=namespace, attributes=normalized, locator=self._locator)
        # catch DTDErrors and decorate them
        except self.DTDError as error:
            error.parser = self
            error.document = self._document
            error.locator = newLocator(self._locator)
            raise

        # push the current node on the stack
        self._nodeStack.append(self._currentNode)
        # and make the new node the focus of attention
        self._currentNode = node

        return


    def characters(self, content):
        """
        Handler for the content of a tag
        """

        if self.ignoreWhitespace:
            content = content.strip()

        if content:
            # print(
                # "line {0}, col {1}: characters {2!r}".format(
                    # self._locator.getLineNumber(), self._locator.getColumnNumber(), content))
            try:
                # get the current node handler to process the element content
                self._currentNode.content(text=content, locator=self._locator)
            except AttributeError as error:
                # raise an error if it doesn't have one
                msg = (
                    "element {0._currentNode._pyre_tag!r} does not accept character data"
                    .format(self))
                raise self.DTDError(
                    parser=self, document=self._document,
                    description=msg, locator=self._locator) from error

        return


    def endElement(self, name):
        """
        Handler for the end of an element
        """
        # print(
            # "line {0}, col {1}: end element {2!r}".format(
                # self._locator.getLineNumber(), self._locator.getColumnNumber(), name))

        # grab the current node and its parent
        node = self._currentNode
        self._currentNode = self._nodeStack.pop()

        # attempt to
        try:
            # let the parent node know we reached an element end
            node.notify(parent=self._currentNode, locator=self._locator)
        # leave our own exception alone
        except self.ParsingError: raise
        # convert everything else to a ProcessingError
        except Exception as error:
            # the reason
            msg = "error while calling the method 'notify' of {}".format(self._currentNode)
            # the error
            raise self.ProcessingError(
                parser=self, document=self._document,
                description=msg, saxlocator=self._locator) from error

        return


    def endElementNS(self, name, qname):
        """
        Handler for the end of a namespace qualified element

        Currently there is no known reason to process this differently from normal elements,
        since the reader already knows how to get hold of the responsible handler for this
        event
        """
        return self.endElement(name)


    def endDocument(self):
        """
        Handler for the end of the document
        """

        # print(
            # "line {0}, col {1}: end document".format(
                # self._locator.getLineNumber(), self._locator.getColumnNumber()))

        self._document.finalize(locator=self._locator)

        self._nodeStack = []
        self._currentNode = None

        return


# end of file
