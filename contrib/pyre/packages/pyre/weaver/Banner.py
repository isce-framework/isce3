# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access to the pyre package
import pyre


# my interface
from .Stationery import Stationery


class Banner(pyre.component, family="pyre.weaver.layouts.banner", implements=Stationery):
    """
    The base component for content generation
    """

    width = pyre.properties.int(default=100)
    width.doc = "the preferred width of the generated text"

    authors = pyre.properties.strings(default="[{pyre.user.name}]")
    authors.doc = "the name of the entities to blame for this content"

    affiliation = pyre.properties.str(default="{pyre.user.affiliation}")
    affiliation.doc = "the author's institutional affiliation"

    copyright = pyre.properties.str()
    copyright.doc = "the copyright note"

    license = pyre.properties.str()
    license.doc = "the license"

    footer = pyre.properties.str(default='end of file')
    footer.doc = "the marker to drop at the bottom of the document"


# end of file
