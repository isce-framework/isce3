# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from ..patterns.AttributeClassifier import AttributeClassifier


class Templater(AttributeClassifier):
    """
    Metaclass that inspects record declarations and endows their instances with the necessary
    infrastructure to support

    * named access of the fields in a record
    * composition via inheritance
    * derivations, i.e. fields whose values depend on the values of other fields
    """


    # types: the descriptor categories
    from . import field as pyre_field
    from . import measure as pyre_measure
    from . import derivation as pyre_derivation
    from . import literal as pyre_literal
    # the tuples
    from .Mutable import Mutable as pyre_mutableTupleType
    from .Immutable import Immutable as pyre_immutableTupleType
    # my field selector
    from .Selector import Selector as pyre_selector
    # my field value accessor
    from .Accessor import Accessor as pyre_accessor
    # the value extractors
    from .Extractor import Extractor as pyre_extractor # simple immutable tuples
    from .Evaluator import Evaluator as pyre_evaluator # complex immutable tuples
    from .Calculator import Calculator as pyre_calculator # simple mutable tuples
    from .Compiler import Compiler as pyre_compiler # complex mutable tuples


    # meta-methods
    def __new__(cls, name, bases, attributes, id=None, **kwds):
        """
        The builder of a new record class.

        Scans through the attributes of the class record being built and harvests the meta-data
        descriptors. These descriptors are removed for the attribute dictionary and replaced
        later with accessors appropriate for each type of record.
        """
        # make a pile foe the meta-data descriptors
        localFields = []
        # harvest them
        for fieldName, field in cls.pyre_harvest(attributes, cls.pyre_field):
            # bind them
            field.bind(name=fieldName)
            # and add them to the pile
            localFields.append(field)

        # build an attribute to hold the locally declared fields
        attributes["pyre_name"] = name if id is None else id
        attributes["pyre_localFields"] = tuple(localFields)

        # remove the field descriptors; we replace them in {__init__} with selectors
        for field in localFields: del attributes[field.name]

        # build the class record
        record = super().__new__(cls, name, bases, attributes, **kwds)

        # now that the class record is built, we can look for inherited fields as well; we
        # traverse the {__mro__} in reverse order and place inherited fields ahead of local
        # ones; this corresponds to the intuitive layout that users expect. further,
        # derivations are expressions involving any of the fields accessible at the point of
        # their declaration, so all of them must have been populated already

        # initialize the three piles
        fields = []
        measures = []
        derivations = []
        # for each base class
        for base in reversed(record.__mro__):
            # skip the ones that are not records themselves
            if not isinstance(base, cls): continue
            # get all of the locally declared record fields
            for field in base.pyre_localFields:
                # add this to the pile
                fields.append(field)
                # if it is a measure
                if cls.pyre_isMeasure(field):
                    # add it to the measure pile
                    measures.append(field)
                # if it is a derivation
                elif cls.pyre_isDerivation(field):
                    # add it to the pile of derivations
                    derivations.append(field)
                # otherwise
                else:
                    # we have a problem; get the journal
                    import journal
                    # and complain
                    raise journal.firewall('pyre.records').log(
                        'unknown field type: {}'.format(field))

        # attach them to the class record
        record.pyre_fields = tuple(fields)
        # filter the measures
        record.pyre_measures = tuple(measures)
        # and the derivations
        record.pyre_derivations = tuple(derivations)

        # finally, some clients need a map from fields to their index in our underlying tuple
        record.pyre_index = dict((field, index) for index, field in enumerate(fields))

        # show me
        # print("{}:".format(name))
        # print("  fields: {}".format(tuple(field.name for field in record.pyre_fields)))
        # print("  measures: {}".format(tuple(field.name for field in record.pyre_measures)))
        # print("  derivations: {}".format(tuple(field.name for field in record.pyre_derivations)))
        # print("  index:")
        # for field, index in record.pyre_index.items():
            # print("    {.name} -> {}".format(field, index))

        # all done
        return record


    def __init__(self, name, bases, attributes, **kwds):
        """
        Decorate a newly minted record

        Now that the class record is built and all the meta-data have been harvested, we can
        build the generators of my instances. The are two of them: one for immutable instances,
        built using a named tuple whose fields are the actual values of the various fields;
        and one for mutable instances, built from a named tuple whose fields are {pyre.calc}
        nodes.
        """
        # chain up
        super().__init__(name, bases, attributes, **kwds)

        columns = {}
        # add selectors for all my fields we removed in {__new__}
        for index, field in enumerate(self.pyre_fields):
            # map this field to its column number
            columns[field] = index
            # build a selector
            selector = self.pyre_selector(field=field, index=index)
            # attach it
            setattr(self, field.name, selector)
        # attach the column map
        self.pyre_columns = columns

        # build value accessors for the data tuples
        attributes = dict(
            # map the name of the field to an accessor
            (field.name, self.pyre_accessor(field=field, index=index))
            # for each of my fields
            for index, field in enumerate(self.pyre_fields))

        # build the helper classes that generate my instances
        mutable = type('mutable', (self.pyre_mutableTupleType,), attributes)
        immutable = type('immutable', (self.pyre_immutableTupleType,), attributes)

        # if i have derivations
        if self.pyre_derivations:
            # attach the value extraction strategies that are aware of derivations
            mutable.pyre_extract = self.pyre_compiler()
            immutable.pyre_extract = self.pyre_evaluator()
        # otherwise
        else:
            # attach the fast value extraction strategies
            mutable.pyre_extract = self.pyre_calculator()
            immutable.pyre_extract = self.pyre_extractor()

        # mark them as mine
        mutable.pyre_layout = self
        immutable.pyre_layout = self
        # and attach them
        self.pyre_mutableTuple = mutable
        self.pyre_immutableTuple = immutable

        # all done
        return


    # predicates
    @classmethod
    def pyre_isMeasure(cls, field):
        """
        Predicate that tests whether {field} is a measure
        """
        # easy...
        return field.category == 'descriptor'


    @classmethod
    def pyre_isDerivation(cls, field):
        """
        Predicate that tests whether {field} is a derivation
        """
        # easy...
        return field.category == 'operator'


# end of file
