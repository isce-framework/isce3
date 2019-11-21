# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# my metaclass
from .Templater import Templater
# superclass
from ..framework.Dashboard import Dashboard


# declaration
class Record(Dashboard, metaclass=Templater):
    """
    The base class for representing data extracted from persistent stores.

    Records have field descriptors that provide the information necessary to convert data
    between the representation used by the persistent store and the native python object
    required by the application.

    Records are similar to named tuples: the underlying storage mechanism is a tuple, and the
    fields are descriptors that provide named access to the tuple items. They are superior to
    named tuples since they enable the data model designer to specify types and constraints
    that must be satisfied by the data, and automate the conversion process to a large degree.

    Inheritance among {Record} subclasses is interpreted as composition: the set of fields that
    define a record is built out of the descriptors declared both locally and by all of its
    ancestors. Descriptor composition is subject to name shadowing.

    Records support {derivations}: fields whose value is computed using other record
    fields. Such fields are built automatically whenever a field declaration contains any sort
    of arithmetic on the right hand side.

    Details of the current implementation:

    * Storage for the record values is provided by {tuple}. This implies that indexed access
      using integers works as expected and does not require any special handling

    * Named access is handled through the field descriptors. Supporting composition via
      inheritance complicates the implementation a bit, as the rank of a given field is not
      known until the class mro is traversed and shadowing is taken into account. Each {Record}
      subclass maintains its own map from descriptor instances to the integer rank in the
      corresponding underlying tuple.
    """


    # types
    # exceptions
    from ..constraints.exceptions import ConstraintViolationError

    # public data; patched by the metaclass
    pyre_name = None # the name of the record
    # structure
    pyre_localFields = None # the tuple of locally declared record fields
    # the full piles that include inherited entries
    pyre_fields = None # the tuple of all accessible fields, both local and inherited
    pyre_measures = None # the tuple of all primary fields
    pyre_derivations = None # the tuple of fields whose values are computed on the fly
    # a map from field descriptors to their column index
    pyre_columns = None


    # interface; patched by the metaclass
    @classmethod
    def pyre_immutable(cls, data=None, **kwds):
        """
        Build a mutable instance
        """
        # easy enough
        return cls.pyre_immutableTuple(record=cls, data=data, **kwds)


    @classmethod
    def pyre_mutable(cls, data=None, **kwds):
        """
        Build a mutable instance
        """
        # easy enough
        return cls.pyre_mutableTuple(record=cls, data=data, **kwds)


    # support for readers that want to match their headers to my fields
    @classmethod
    def pyre_selectColumns(cls, headers):
        """
        Prepare a tuple of the column numbers needed to populate my instances, given a map
        (column name) -> (column index).

        This enables the managers of the various persistent stores to build record instances
        from a subset of the information they have access to. It is also designed to perform
        column name translations from whatever meta data is available in the store to the
        canonical record field names
        """
        # iterate over my measures
        for measure in cls.pyre_measures:
            # and over its aliases
            for alias in measure.aliases:
                # if this alias appears in the headers
                try:
                    # compute the column index and return it
                    yield headers[alias]
                    # get the next field
                    break
                except KeyError:
                    continue
            # error: unable to find a source for this field
            else:
                # if it is not an optional field
                if not measure.optional:
                    # complain
                    msg = "unable to find a source for field {!r}".format(measure.name)
                    import journal
                    raise journal.error("pyre.records").log(msg)
        # all done
        return


# end of file
