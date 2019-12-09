# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import pyre


# declaration
class Stationery(pyre.protocol, family="pyre.weaver.layouts"):
    """
    The protocol that layout strategies must implement
    """


    # traits
    width = pyre.properties.int()
    width.doc = "the preferred width of the generated text"

    authors = pyre.properties.strings()
    authors.doc = "the name of the entities to blame for this content"

    affiliation = pyre.properties.str()
    affiliation.doc = "the author's institutional affiliation"

    copyright = pyre.properties.str()
    copyright.doc = "the copyright notice"

    license = pyre.properties.str()
    license.doc = "the license"

    footer = pyre.properties.str()
    footer.doc = "the marker to drop at the bottom of the document"


    # utilities
    @classmethod
    def pyre_default(cls, **kwds):
        """
        Choose a layout as the default
        """
        # the current default is {Banner}
        from .Banner import Banner
        return Banner


# end of file
