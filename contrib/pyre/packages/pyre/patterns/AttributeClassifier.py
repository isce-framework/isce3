# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import collections
# superclass
from .AbstractMetaclass import AbstractMetaclass


# class declaration
class AttributeClassifier(AbstractMetaclass):
    """
    A base metaclass that enables attribute categorization.

    A common pattern in pyre is to define classes that contain special attributes whose purpose
    is to collect declaration meta data and associate them with a class attribute. These
    attributes are processed by metaclasses and are converted into appropriate behavior. For
    example, components have properties, which are decorated descriptors that enable external
    configuration of component state. Similarly, XML parsing happens with the aid of classes
    that capture the syntax, semantics and processing behavior of tags by employing descriptors
    to capture the layout of an XML document.

    This class defines {pyre_harvest}, which scans the class attribute dictionary for instances
    of the special class {descriptor}. It also overrides {__prepare__} to provide attribute
    storage that records the order in which attributes were encountered in the class record.
    """


    # data
    pyre_reserved = set()


    # meta methods
    @classmethod
    def __prepare__(cls, name, bases, **kwds):
        """
        Build an attribute table that maintains a category index for attribute descriptors
        """
        # use an ordered dictionary
        return collections.OrderedDict()


    # interface
    @classmethod
    def pyre_harvest(cls, attributes, descriptor):
        """
        Examine {attributes}, looking for instances of {descriptor}
        """
        # reserved names are excluded from harvesting
        reserved = cls.pyre_reserved
        # loop over the attributes
        for name, attribute in attributes.items():
            # if this is a descriptor that's not in the reserved list
            if isinstance(attribute, descriptor) and name not in reserved:
                # return it to the caller along with its name
                yield name, attribute
        # all done
        return


# end of file
