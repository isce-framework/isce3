# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import pyre


# declaration
class Weaver(pyre.component, family="pyre.weaver"):
    """
    The base component for content generation
    """

    # types
    # protocols for my traits
    from .Language import Language


    # public state
    language = Language(default=None)
    language.doc = "the desired output language"


    # public interface
    @pyre.export
    def weave(self, **kwds):
        """
        Assemble the {document}
        """
        # let my language do its thing
        yield from self.language.render(**kwds)
        # and return
        return


# end of file
