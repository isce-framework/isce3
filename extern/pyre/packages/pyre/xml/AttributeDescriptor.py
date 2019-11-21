# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


from .Descriptor import Descriptor


class AttributeDescriptor(Descriptor):
    """
    This class serves as the resting place for element metadata provided by the user during
    DTD formation. It is used by DTD-derived metaclasses to decorate the handlers of the various
    XML elements

    This capability is not yet fully developed.
    """


    # attribute metadata
    name = None # set by Named

    # the attriibute type
    # may be one of CDATA, ID, IDREFS, NMTOKEN, NMTOKENS, ENTITY, ENTITIES, NOTATION, XML
    # or a tuple of valid choices; see pyre.xml.enumerated()
    type = None

    # attribute requirements:
    #   pyre.xml.IMPLIED:
    #     the attribute is optional
    #   pyre.xml.REQUIRED:
    #     the attribute is required
    #     signal an error if the element does not specify a value
    #   pyre.xml.FIXED:
    #     the default value is the only possible value for the attribute
    #     signal an error if the document contains anything else
    presence = None

    # the default value
    default = None # the default value for the attribute


# end of file
