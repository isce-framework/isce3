# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
This package contains the machinery to build parsers of XML documents

The goal of this package is to enable context-specific processing of the information content of
the XML file. It is designed to provide support for applications that use XML files as an
information exchange mechanism and consider parsing the XML file as another means for
decorating their data structures.

There are two styles of processing the contents of XML files in common use that go by the names
DOM and SAX. DOM processing is generally considered to be the simplest way. It constructs a
representation of the entire file with a single function call to the parser and returns a data
structure that has a very faithful representation of the layout of the file, along with a rich
interface for navigating through the representation and discovering its content. In contrast,
SAX parsing is event driven. The parser scans through the document and generates events for
each significant encounter with the document contents, such the opening or closing of an XML
tag, or encountering data in the body of a tag. The client interacts with the parser by
registering handlers for each type of event that are responsible for absorbing the information
collected by te parser. This trades some complexity in the handling of the document for the
savings of not needing to build and subsequently navigate through an intermediate data
structure.

Overall, SAX style document parsing should be faster and more space efficient for most uses. It
also opens up the possibility that an application can implement its event handlers in a way that
directly decorate its own internal data structures. The purpose of this package is to help
minimize the code complexity required to craft event handlers and register them with the
parser.

As a prototypical example, consider an address book application that maintains contact
information for friends and clients. Such an application is likely to have a rather extensive
set of classes that handle the various type of contacts and their associated information. The
addressbook is likely some type of container of such classes. Support for retrieving contact
information from XML files ideally should be just another way to construct the application
specific data structures and adding them to the addressbook container. The application
developer should be shielded as much as possible from the mechanics of parsing XML files and
the intermediate data structures necessary to store the file contents.

The application in the examples directory is a trivial implementation of an addressbook but
highlights all the important steps for creating handlers/decorators for converting XML files
into live data structures.
"""

# externals
import pyre.tracking


# factories
def newReader(**kwds):
    """
    Create a new reader
    """
    from .Reader import Reader
    return Reader(**kwds)


def newLocator(saxlocator):
    """
    Convert a locator from the SAX parser to a pyre.tracking.FileLocator
    """
    return pyre.tracking.file(
        source=saxlocator.getSystemId(),
        line=saxlocator.getLineNumber(), column=saxlocator.getColumnNumber())


# support for document descriptors
from .Node import Node as node
from .Ignorable import Ignorable as ignorable
from .Document import Document as document
from .ElementDescriptor import ElementDescriptor as element


# package constants
# attribute types
CDATA = "CDATA"
ID = "ID"
IDREFS = "IDREFS"
NMTOKEN = "NMTOKEN"
NMTOKENS = "NMTOKENS"
ENTITY = "ENTITY"
ENTITIES = "ENTITIES"
NOTATION = "NOTATION"
XML = "xml:"

# default value types
REQUIRED = "#REQUIRED"
IMPLIED = "#IMPLIED"
FIXED = "#FIXED"


# end of file
