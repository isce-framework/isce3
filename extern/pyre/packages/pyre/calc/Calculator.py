# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .. import algebraic


# declaration
class Calculator(algebraic.algebra):
    """
    Metaclass that grants nodes value management capabilities
    """


    # types
    from .Datum import Datum as base
    # entities: my literals, variables, operators and sequences have values
    from .Const import Const as const
    from .Value import Value as value
    from .Evaluator import Evaluator as evaluator
    from .Mapping import Mapping as mapping
    from .Sequence import Sequence as sequence
    # the new types of entities that support evaluation after name resolution
    from .Expression import Expression as expression
    from .Interpolation import Interpolation as interpolation
    from .Unresolved import Unresolved as unresolved
    # references to other nodes
    from .Reference import Reference as reference
    # local operators
    from .Average import Average as average
    from .Count import Count as count
    from .Maximum import Maximum as maximum
    from .Minimum import Minimum as minimum
    from .Product import Product as product
    from .Sum import Sum as sum
    # value change notification
    from .Reactor import Reactor as reactor
    from .Observer import Observer as observer
    from .Observable import Observable as observable
    from .Dependent import Dependent as dependent
    from .Dependency import Dependency as dependency
    # value processing
    from .Preprocessor import Preprocessor as preprocessor
    from .Postprocessor import Postprocessor as postprocessor
    # value memoization
    from .Memo import Memo as memo
    # value filtering
    from .Filter import Filter as filter


    # meta-methods
    def __new__(cls, name, bases, attributes, ignore=False, **kwds):
        """
        Build a new class record
        """
        # build the record
        record = super().__new__(cls, name, bases, attributes, ignore=ignore, **kwds)
        # for specially marked classes, we are all done
        if ignore or cls.isIgnorable(bases): return record

        # the rest get some extra decoration: expressions, interpolations, and references
        # build the list of base classes for expression
        derivation = tuple(cls.expressionDerivation(record))
        # make one
        record.expression = cls('expression', derivation, {}, ignore=True)

        # build the list of base classes for interpolation
        derivation = tuple(cls.interpolationDerivation(record))
        # make one
        record.interpolation = cls('interpolation', derivation, {}, ignore=True)

        # build the list of base classes for interpolation
        derivation = tuple(cls.sequenceDerivation(record))
        # make one
        record.sequence = cls('sequence', derivation, {}, ignore=True)

        # build the list of base classes for interpolation
        derivation = tuple(cls.mappingDerivation(record))
        # make one
        record.mapping = cls('mapping', derivation, {}, ignore=True)

        # build the list of base classes for reference
        derivation = tuple(cls.referenceDerivation(record))
        # make one
        record.reference = cls('reference', derivation, {}, ignore=True)

        # build the list of base classes for unresolved
        derivation = tuple(cls.unresolvedDerivation(record))
        # make one
        record.unresolved = cls('unresolved', derivation, {}, ignore=True)

        # build the list of base classes for average
        derivation = tuple(cls.managedCompositeDerivation(cls.average, record))
        # make one
        record.average = cls('average', derivation, {}, ignore=True)

        # build the list of base classes for count
        derivation = tuple(cls.managedCompositeDerivation(cls.count, record))
        # make one
        record.count = cls('count', derivation, {}, ignore=True)

        # build the list of base classes for max
        derivation = tuple(cls.managedCompositeDerivation(cls.maximum, record))
        # make one
        record.max = cls('max', derivation, {}, ignore=True)

        # build the list of base classes for min
        derivation = tuple(cls.managedCompositeDerivation(cls.minimum, record))
        # make one
        record.min = cls('min', derivation, {}, ignore=True)

        # build the list of base classes for product
        derivation = tuple(cls.managedCompositeDerivation(cls.product, record))
        # make one
        record.product = cls('product', derivation, {}, ignore=True)

        # build the list of base classes for sum
        derivation = tuple(cls.managedCompositeDerivation(cls.sum, record))
        # make one
        record.sum = cls('sum', derivation, {}, ignore=True)

        # all done
        return record


    # implementation details
    @classmethod
    def literalDerivation(cls, record):
        """
        Contribute to the list of ancestors of the representation of literals
        """
        # my literals have const values
        yield cls.const
        # they are observable
        yield cls.observable
        # and whatever else my superclass says
        yield from super().literalDerivation(record)
        # all done
        return


    @classmethod
    def variableDerivation(cls, record):
        """
        Contribute to the list of ancestors of the representation of literals
        """
        # my variable may reject invalid input
        yield cls.filter
        # make a managed dependent
        yield from cls.managedDependencyDerivation()
        # my variables have values
        yield cls.value
        # and whatever else my superclass says
        yield from super().variableDerivation(record)
        # all done
        return


    @classmethod
    def operatorDerivation(cls, record):
        """
        Contribute to the list of ancestors of the representation of operators
        """
        # make a managed dependent
        yield from cls.managedDependentDerivation()
        # my operators know how to compute their values
        yield cls.evaluator
        # and whatever else my superclass says
        yield from super().operatorDerivation(record)
        # all done
        return


    @classmethod
    def sequenceDerivation(cls, record):
        """
        Contribute to the list of ancestors of the representation of operators
        """
        # make a managed dependent
        yield from cls.managedDependentDerivation()
        # if the record has anything to say
        if record.sequence:
            # this is its spot
            yield record.sequence
        # my sequences know how to compute their values
        yield cls.sequence
        # and whatever else my superclass says
        yield from cls.compositeDerivation(record)
        # all done
        return


    @classmethod
    def mappingDerivation(cls, record):
        """
        Contribute to the list of ancestors of the representation of operators
        """
        # make a managed dependent
        yield from cls.managedDependentDerivation()
        # if the record has anything to say
        if record.mapping:
            # this is its spot
            yield record.mapping
        # my mappings know how to compute their values
        yield cls.mapping
        # and whatever else my superclass says
        yield from cls.compositeDerivation(record)
        # all done
        return


    @classmethod
    def expressionDerivation(cls, record):
        """
        Contribute to the list of ancestors of the representation of expressions
        """
        # make a managed dependent
        yield from cls.managedDependentDerivation()
        # if the record has anything to say
        if record.expression:
            # this is its spot
            yield record.expression
        # this where they fit
        yield cls.expression
        # and whatever else my superclass says
        yield from cls.compositeDerivation(record)
        # all done
        return


    @classmethod
    def interpolationDerivation(cls, record):
        """
        Contribute to the list of ancestors of the representation of interpolations
        """
        # make a managed dependent
        yield from cls.managedDependentDerivation()
        # if the record has anything to say
        if record.interpolation:
            # this is its spot
            yield record.interpolation
        # this where they fit
        yield cls.interpolation
        # and whatever else my superclass says
        yield from cls.compositeDerivation(record)
        # all done
        return


    @classmethod
    def referenceDerivation(cls, record):
        """
        Contribute to the list of ancestors of the representation of references
        """
        # make a managed dependent
        yield from cls.managedDependentDerivation()
        # if the record has anything to say
        if record.reference:
            # this is its spot
            yield record.reference
        # this where they fit
        yield cls.reference
        # and whatever else my superclass says
        yield from cls.compositeDerivation(record)
        # all done
        return


    @classmethod
    def unresolvedDerivation(cls, record):
        """
        Contribute to the list of ancestors of the representation of unresolved nodes
        """
        # my unresolved nodes are observable
        yield cls.observable
        # if the record has anything to say
        if record.unresolved: yield record.unresolved
        # my unresolved nodes know how to compute their values
        yield cls.unresolved
        # and whatever else my superclass says
        yield from cls.leafDerivation(record)
        # all done
        return


    @classmethod
    def managedDependencyDerivation(cls):
        """
        Build the canonical derivation of node that other nodes can depend on
        """
        # my local nodes memoize their values
        yield cls.memo
        # support arbitrary value conversions
        yield cls.preprocessor
        yield cls.postprocessor
        # notify their clients of changes to their values and respond when the values of their
        # operands change
        yield cls.dependency
        yield cls.observable
        # all done
        return


    @classmethod
    def managedDependentDerivation(cls):
        """
        Place {dependent} in the right spot in the inheritance graph of {record}
        """
        # my local nodes memoize their values
        yield cls.memo
        # support arbitrary value conversions
        yield cls.preprocessor
        yield cls.postprocessor
        # notify their clients of changes to their values and respond when the values of their
        # operands change
        yield cls.dependent
        yield cls.observer
        yield cls.dependency
        yield cls.observable
        # all done
        return


    @classmethod
    def managedCompositeDerivation(cls, composite, record):
        """
        Place the class {composite} in the right spot in the {record} inheritance graph
        """
        # make a managed dependent
        yield from cls.managedDependentDerivation()
        # this where {composite} fits
        yield composite
        # and whatever else my superclass says
        yield from cls.compositeDerivation(record)
        # all done
        return


# end of file
