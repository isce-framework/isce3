# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import collections
# my base class is from {pyre.schemata}
from ..schemata.Schema import Schema


# declaration
class Typed(Schema):
    """
    Mix-in class that encapsulates type information. Its instances participate in value
    conversions from external representations to python internal forms.
    """


    # public data
    # value preprocessors
    converters = ()
    # value post processors
    normalizers = ()
    # consistency checks
    validators = ()


    # interface
    def process(self, value, **kwds):
        """
        Walk {value} through the steps from raw to validated
        """
        # {None} is special; leave it alone
        if value is None: return None
        # so are string representations of {None}
        if isinstance(value, str) and value.strip().lower() == "none": return None

        # otherwise, convert
        for converter in self.converters:
            # by asking each register converter to prep the value
            value = converter(descriptor=self, value=value, **kwds)
        # cast
        value = self.coerce(value=value, **kwds)
        # normalize
        for normalizer in self.normalizers:
            # by asking each normalizer to bring {value} in normal form
            value = normalizer(descriptor=self, value=value, **kwds)
        # validate
        for validator in self.validators:
            # by giving a chance to each validator to raise an exception
            value = validator(descriptor=self, value=value, **kwds)
        # and return the new value
        return value


    # framework requests
    def bind(self, **kwds):
        """
        Called by my client to let me know that all available meta-data have been harvested
        """
        # for convenience, clients are allowed to declare values processors in somewhat free
        # form; repair this usage that breaks my representation constraints by making sure my
        # value processors are stored in a mutable and iterable container
        self.converters = self.listify(self.converters)
        self.normalizers = self.listify(self.normalizers)
        self.validators = self.listify(self.validators)

        # chain up
        return super().bind(**kwds)


    # meta methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)

        # initialize my value processors to something mutable and order retaining
        self.converters = []
        self.normalizers = []
        self.validators = []

        # all done
        return


    # implementation details
    def listify(self, processors):
        """
        Make sure {processors} is an iterable regardless of what the user left behind
        """
        # handle anything empty
        if not processors: return []
        # if i have an iterable
        if isinstance(processors, collections.Iterable):
            # turn it into a list
            return list(processors)
        # otherwise, place the lone processor in a list
        return [processors]


# end of file
