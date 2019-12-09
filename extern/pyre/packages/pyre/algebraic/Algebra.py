# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from ..patterns.AbstractMetaclass import AbstractMetaclass


# declaration
class Algebra(AbstractMetaclass):
    """
    Metaclass that endows its instances with algebraic structure
    """


    # types
    # structural
    from .Leaf import Leaf as leaf
    from .Composite import Composite as composite
    # algebraic
    from .Arithmetic import Arithmetic as arithmetic
    from .Ordering import Ordering as ordering
    from .Boolean import Boolean as boolean
    # the base node
    from .AbstractNode import AbstractNode as base
    # nodes
    from .Literal import Literal as literal
    from .Operator import Operator as operator
    from .Variable import Variable as variable



    # meta-methods
    def __new__(cls, name, bases, attributes,
                arithmetic=True, ordering=True, boolean=True,
                ignore=False,
                **kwds):
        """
        Build a new class record
        """
        # specially marked classes
        if ignore or cls.isIgnorable(bases):
            # bypass any of my processing
            return super().__new__(cls, name, bases, attributes, **kwds)

        # prime the list of ancestors
        derivation = [cls.base]
        # if we were asked to support arithmetic, add support for it
        if arithmetic: derivation.append(cls.arithmetic)
        # if we were asked to support ordering, add support for it
        if ordering: derivation.append(cls.ordering)
        # if we were asked to support boolean operations, add support for it
        if boolean: derivation.append(cls.boolean)
        # wrap up by piling on the actual bases of the client
        bases = tuple(derivation) + bases

        # build the record
        record = super().__new__(cls, name, bases, attributes, **kwds)

        # build the list of base classes for the literal
        derivation = tuple(cls.literalDerivation(record))
        # make one
        record.literal = cls('literal', derivation, {}, ignore=True)

        # build the list of base classes for the variable
        derivation = tuple(cls.variableDerivation(record))
        # make one
        record.variable = cls('variable', derivation, {}, ignore=True)

        # build the list of base classes for operators
        derivation = tuple(cls.operatorDerivation(record))
        # make one
        record.operator = cls('operator', derivation, {}, ignore=True)

        # mark it
        record._pyre_hasAlgebra = True

        # return the record
        return record


    # implementation details
    @classmethod
    def leafDerivation(cls, record):
        """
        Contribute to the list of ancestors of the representation of literals
        """
        # if the {record} specifies a leaf mix-in, add it to the pile
        if record.leaf: yield record.leaf
        # yield the default leaf class
        yield cls.leaf
        # and the buck stops here...
        yield record
        # all done
        return


    @classmethod
    def compositeDerivation(cls, record):
        """
        Contribute to the list of ancestors of the representation of literals
        """
        # if the {record} specifies a composite mix-in, add it to the pile
        if record.composite: yield record.composite
        # yield the default composite class
        yield cls.composite
        # and the buck stops here...
        yield record
        # all done
        return


    @classmethod
    def literalDerivation(cls, record):
        """
        Contribute to the list of ancestors of the representation of literals
        """
        # if the class record specifies a literal mix-in use it
        if record.literal: yield record.literal
        # must also derive from the default
        yield cls.literal
        # get the classes necessary to make leaves
        yield from cls.leafDerivation(record)
        # all done
        return


    @classmethod
    def operatorDerivation(cls, record):
        """
        Contribute to the list of ancestors of the representation of operators
        """
        # if the class record specifies a operator mix-in use it
        if record.operator: yield record.operator
        # must also derive from the default
        yield cls.operator
        # get the classes necessary to make composites
        yield from cls.compositeDerivation(record)
        # all done
        return


    @classmethod
    def variableDerivation(cls, record):
        """
        Contribute to the list of ancestors of the representation of variables
        """
        # if the class record specifies a variable mix-in use it
        if record.variable: yield record.variable
        # must also derive from the default
        yield cls.variable
        # get the classes necessary to make leaves
        yield from cls.leafDerivation(record)
        # all done
        return


    @classmethod
    def isIgnorable(cls, bases):
        """
        Filter that determines whether a class should be decorated or not.

        This is necessary because the metaclass is asked to process all subclasses of the type
        that injected it in the hierarchy. In our case, variables, operators and the like would
        also pass through the process. This routine detects these cases and avoids them
        """
        # go through each of the bases
        for base in bases:
            # looking for
            try:
                # a marked one
                base._pyre_hasAlgebra
            # of that fails
            except AttributeError:
                # perfect; check the next one
                continue
            # otherwise
            else:
                # this class derived from one of mine
                return base._pyre_hasAlgebra
        # all good
        return False


# end of file
