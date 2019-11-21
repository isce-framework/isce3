# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Preprocessor:
    """
    A mix-in class that performs arbitrary transformations on the value of a node
    """


    # types
    from ..schemata import identity
    from .exceptions import EvaluationError

    # public data
    preprocessor = None


    # value management
    def setValue(self, value, **kwds):
        """
        Hand the incoming {value} to my {preprocessor} before storing it
        """
        # attempt to
        try:
            # process it
            value = self.preprocessor(value=value, node=self, **kwds)
        # protect against framework bugs: asking for configurable attributes that don't exist
        except AttributeError as error:
            # get the journal
            import journal
            # complain
            raise journal.firewall('pyre.calc').log(str(error))

        # and return it
        return super().setValue(value=value, **kwds)


    # meta-methods
    def __init__(self, preprocessor=identity().coerce, **kwds):
        # chain up
        super().__init__(**kwds)
        # set my value processor
        self.preprocessor = preprocessor
        # all done
        return


# end of file
