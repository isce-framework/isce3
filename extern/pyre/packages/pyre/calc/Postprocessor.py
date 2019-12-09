# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Postprocessor:
    """
    A mix-in class that performs arbitrary transformations on the value of a node
    """


    # types
    from ..schemata import identity
    from .exceptions import EvaluationError

    # public data
    @property
    def postprocessor(self):
        """
        Read access to my value post processor
        """
        # easy enough
        return self._postprocessor

    @postprocessor.setter
    def postprocessor(self, processor):
        """
        Write access to my postprocessor
        """
        # set it
        self._postprocessor = processor
        # invalidate my value cache
        self.flush()
        # all done
        return


    # value management
    def getValue(self, **kwds):
        """
        Intercept the node value retriever and make sure that the value the caller gets has
        been through my {postprocessor}
        """
        # get the value
        value = super().getValue()
        # attempt to
        try:
            # process it
            value = self.postprocessor(value=value, node=self, **kwds)
        # protect against framework bugs: asking for configurable attributes that don't exist
        except AttributeError as error:
            # get the journal
            import journal
            # complain
            raise journal.firewall('pyre.calc').log(str(error))

        # and return it
        return value


    # meta-methods
    def __init__(self, postprocessor=identity().coerce, **kwds):
        # chain up
        super().__init__(**kwds)
        # set my value processor
        self._postprocessor = postprocessor
        # all done
        return


    # private data
    _postprocessor = None


# end of file
