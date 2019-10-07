# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# unary expressions
class UnaryPostfix:
    """
    The base class for postfix unary expression factories
    """

    # public data
    operand = None
    operator = None


    # interface
    def sql(self):
        """SQL rendering of the expression I represent"""
        # straightforward
        return "{} {.operator}".format(self.operand.sql(), self)

    # meta-methods
    def __init__(self, operand, **kwds):
        # chain up
        super().__init__(**kwds)
        # store my value
        self.operand = operand
        # all done
        return


class IsNull(UnaryPostfix):
    """
    A node factory that takes a field reference {op} and builds the expression {op IS NULL}
    """
    # public data
    operator = "IS NULL"


class IsNotNull(UnaryPostfix):
    """
    A node factory that takes a field reference {op} and builds the expression {op IS NOT NULL}
    """
    # public data
    operator = "IS NOT NULL"


 # some built-in functions
class Cast:
    """
    Implementation of the {CAST} expression
    """

    # interface
    def sql(self, **kwds):
        """
        SQL rendering of the expression I represent
        """
        # easy enough
        return "CAST({} AS {})".format(self.field.sql(**kwds), self.targetType.decl)

    # meta-methods
    def __init__(self, field, targetType, **kwds):
        # chain up
        super().__init__(**kwds)
        # save my field reference
        self.field = field
        # and the target type
        self.targetType = targetType
        # all done
        return


# some built-in operators
class Like:
    """
    Implementation of the LIKE operator
    """

    # interface
    def sql(self, **kwds):
        """
        SQL rendering of the expression I represent
        """
        # easy enough
        return "({} LIKE '{}')".format(self.field.sql(**kwds), self.regex)

    # meta-methods
    def __init__(self, field, regex, **kwds):
        # chain up
        super().__init__(**kwds)
        # save my field reference
        self.field = field
        # and the target type
        self.regex = regex
        # all done
        return


# end of file
