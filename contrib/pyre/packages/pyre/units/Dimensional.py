# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import collections, operator


# declaration
class Dimensional:
    """
    This class comprises the fundamental representation of quantities with units
    """


    # exceptions
    from .exceptions import CompatibilityError, ConversionError


    # public data
    # representational choices
    fundamental = ('kg', 'm', 's', 'A', 'K', 'mol', 'cd') # the SI fundamental units
    zero = (0,) * len(fundamental)
    # default values
    value = 0
    derivation = zero


    # interface
    def isCompatible(self, other):
        """
        Predicate that checks whether {other} has the same derivation as I do
        """
        # attempt to
        try:
            # check for matching derivations
            return self.derivation == other.derivation
        # if {other} is not a dimensional
        except AttributeError:
            # return false
            return self.derivation == self.zero


    # meta methods
    def __init__(self, value, derivation):
        """
        Constructor:
            {value}: the magnitude
            {derivation}: a tuple with the exponents of the fundamental SI units
        """
        self.value = value
        self.derivation = derivation
        return


    # addition
    def __add__(self, other):
        """
        Addition
        """
        # if {other} is not dimensional
        if not isinstance(other, Dimensional):
            # describe the error
            msg = "unsupported operand types for +: {.__name__!r} and {.__name__!r}".format(
                type(self), type(other))
            # report it
            raise TypeError(msg)
        # if the two quantities are not compatible
        if not self.derivation == other.derivation:
            # report an error
            raise self.CompatibilityError(operation="addition", op1=self, op2=other)
        # otherwise compute the result and return it
        return Dimensional(value=self.value+other.value, derivation=self.derivation)


    # subtraction
    def __sub__(self, other):
        """
        Subtraction
        """
        # if {other} is not dimensional
        if not isinstance(other, Dimensional):
            # describe the error
            msg = "unsupported operand types for -: {.__name__!r} and {.__name__!r}".format(
                type(self), type(other))
            # report it
            raise TypeError(msg)
        # if the two quantities are not compatible
        if not self.derivation == other.derivation:
            # report an error
            raise self.CompatibilityError(operation="addition", op1=self, op2=other)
        # otherwise compute the result and return it
        return Dimensional(value=self.value-other.value, derivation=self.derivation)


    # multiplication
    def __mul__(self, other):
        """
        Multiplication
        """
        # if {other} is iterable
        if isinstance(other, collections.abc.Iterable):
            # dispatch the operation to the individual entries
            return type(other)(self*entry for entry in other)

        # otherwise,  get my value
        value = self.value
        # attempt to interpret {other} as a dimensional
        try:
            value *= other.value
        except AttributeError:
            # the only legal alternative is that {other} is a numeric type
            try:
                # multiply
                value *= other
            # and if this fails too
            except TypeError:
                # report an error
                raise self.CompatibilityError(operation="multiplication", op1=self, op2=other)
            # if i am dimensionless, just return the value
            if self.derivation == self.zero: return value
            # otherwise, return a new dimensional
            return Dimensional(value=value, derivation=self.derivation)
        # otherwise, compute the units
        derivation = tuple(map(operator.add, self.derivation, other.derivation))
        # check whether the units canceled
        if derivation == self.zero: return value
        # otherwise build a new one and return it
        return Dimensional(value, derivation)


    # division
    def __truediv__(self, other):
        """
        True division
        """
        # get my value
        value = self.value
        # attempt to interpret {other} as a dimensional
        try:
            value /= other.value
        except AttributeError:
            # the only legal alternative is that {other} is a numaric type
            try:
                # divide
                value /= other
            # if this fails too
            except TypeError:
                # report an error
                raise self.CompatibilityError(operation="division", op1=self, op2=other)
            # if i am dimensionless, just return the value
            if self.derivation == self.zero: return value
            # otherwise, build a dimensional and return it
            return Dimensional(value=value, derivation=self.derivation)
        # otherwise compute the units
        derivation = tuple(map(operator.sub, self.derivation, other.derivation))
        # check whether the units canceled
        if derivation == self.zero: return value
        # and return a new dimensional
        return Dimensional(value=value, derivation=derivation)


    # exponentiation
    def __pow__(self, other):
        """
        Exponentiation
        """
        # compute the magnitude
        value = self.value ** other
        # compute the dimensions
        derivation = tuple(map(operator.mul, [other]*7, self.derivation))
        # build a new dimensional and return it
        return Dimensional(value=value, derivation=derivation)


    # unary plus
    def __pos__(self):
        """
        Unary plus
        """
        # not much to do
        return self


    # unary minus
    def __neg__(self):
        """
        Unary minus
        """
        # return a new one with the value sign reversed
        return Dimensional(value=-self.value, derivation=self.derivation)


    # absolute value
    def __abs__(self):
        """
        Absolute value
        """
        # build a new one with positive value
        return Dimensional(value=abs(self.value), derivation=self.derivation)


    # right multiplication
    def __rmul__(self, other):
        """
        Right multiplication
        """
        # if other is iterable
        if isinstance(other, collections.abc.Iterable):
            # assume it is an iterable of numeric types; dispatch the operation to the
            # individual entries
            return type(other)(entry*self for entry in other)
        # the only other thing i can do is interpret {other} as a numeric type
        value = self.value * other
        # build a new one and return it
        return Dimensional(value=value, derivation=self.derivation)


    # right division
    def __rtruediv__(self, other):
        """
        Right division
        """
        # if other is iterable
        if isinstance(other, collections.abc.Iterable):
            # assume it is an iterable of numeric types; dispatch the operation to the
            # individual entries
            return type(other)(entry/self for entry in other)
        # interpret {other} as a numeric type
        value = other / self.value
        # compute the dimensions
        derivation = tuple(map(operator.neg, self.derivation))
        # build a new one and return it
        return Dimensional(value, derivation)


    # coercion to float
    def __float__(self):
        """
        Conversion to float
        """
        # if i happen to be a disguised float, convert me
        if self.derivation == self.zero:
            # must cast explicitly because an actual float is expected
            return float(self.value)
        # otherwise
        raise self.ConversionError(operand=self)


    # ordering
    def __lt__(self, other):
        """
        Ordering: less than
        """
        # assuming {other} is dimensional
        try:
            # check the dimensions
            if self.derivation == other.derivation:
                # and the values
                return self.value < other.value
        # if not
        except AttributeError:
            # check whether I am dimensionless
            if self.derivation == self.zero:
                # in which case, just compare my value
                return self.value < other
        # the operation is illegal
        raise self.CompatibilityError(operation="<", op1=self, op2=other)


    def __le__(self, other):
        """
        Ordering: less than or equal to
        """
        # assuming {other} is dimensional
        try:
            # check the dimensions
            if self.derivation == other.derivation:
                # and the values
                return self.value <= other.value
        # if not
        # except AttributeError: pass
        except AttributeError:
            # check whether I am dimensionless
            if self.derivation == self.zero:
                # in which case, just compare my value
                return self.value <= other
        # the operation is illegal
        raise self.CompatibilityError(operation="<=", op1=self, op2=other)


    def __eq__(self, other):
        """
        Ordering: equality
        """
        # assuming {other} is dimensional
        try:
            # check the dimensions
            if self.derivation == other.derivation:
                # and the values
                return self.value == other.value
        # if not
        except AttributeError:
            # check whether I am dimensionless
            if self.derivation == self.zero:
                # in which case, just compare my value
                return self.value == other
        # in every other case, the operation is illegal
        raise self.CompatibilityError(operation="==", op1=self, op2=other)


    def __ne__(self, other):
        """
        Ordering: not equal to
        """
        # assuming {other} is dimensional
        try:
            # check the dimensions
            if self.derivation == other.derivation:
                # and the values
                return self.value != other.value
        # if not
        except AttributeError:
            # check whether I am dimensionless
            if self.derivation == self.zero:
                # in which case, just compare my value
                return self.value != other
        # the operation is illegal
        raise self.CompatibilityError(operation="!=", op1=self, op2=other)


    def __gt__(self, other):
        """
        Ordering: greater than
        """
        # assuming {other} is dimensional
        try:
            # check the dimensions
            if self.derivation == other.derivation:
                # and the values
                return self.value > other.value
        # if not
        except AttributeError:
            # check whether I am dimensionless
            if self.derivation == self.zero:
                # in which case, just compare my value
                return self.value > other
        # the operation is illegal
        raise self.CompatibilityError(operation=">", op1=self, op2=other)


    def __ge__(self, other):
        """
        Ordering: greater than or equal to
        """
        # assuming {other} is dimensional
        try:
            # check the dimensions
            if self.derivation == other.derivation:
                # and the values
                return self.value >= other.value
        # if not
        except AttributeError:
            # check whether I am dimensionless
            if self.derivation == self.zero:
                # in which case, just compare my value
                return self.value >= other
        # the operation is illegal
        raise self.CompatibilityError(operation=">=", op1=self, op2=other)


    def __str__(self):
        """
        Conversion to str
        """
        # render my derivation
        derivation = self._strDerivation()
        # if I have units
        if derivation:
            # render my value and my derivation
            return str(self.value) + '*' + self._strDerivation()
        # otherwise, I am dimensionless do just render my value
        return str(self.value)


    def __format__(self, code):
        """
        Formatting support

        The parameter {code} is a string of the form
            value={format_spec},base={scale},label={label}
        where
            {format_spec}: a format specification appropriate for representing floats
            {scale}: a dimensional quantity to be used as a scale for the value
            {label}: the label with units that should follow the magnitude of the quantity

        Example:
            >>> from pyre.units.SI import m,s
            >>> g = 9.81*m/s
            >>> "{accel:value=.2f,base={scale},label=g}".format(accel=100*m/s**2, scale=g)
            '10.2 g'
        """
        # establish the formatting defaults
        fields = {
            'value': '',
            'base': Dimensional(value=1, derivation=self.derivation),
            'label': self._strDerivation(),
            }
        # if the user supplied a format specifier
        if code:
            # assume pretty output
            pretty = True
            # update the formatting fields
            fields.update(field.strip().split('=') for field in code.split(','))
        # otherwise
        else:
            # render in a way recognizable by the parser
            pretty = False

        # get the fields
        value = fields['value']
        base = fields['base']
        label = fields['label']
        # convert the base specification if necessary
        if isinstance(base, str):
            # get the parser factory
            from . import parser
            # access the singleton
            p = parser()
            # make the conversion
            base = p.parse(base)
        # compute the numeric part
        magnitude = self/base
        # if the dimensions label is empty
        if not label:
            # render my value
            return format(self/base, value)
        # otherwise, we have a label; attempt
        try:
            # extract the singular and plural forms
            singular, plural = label.split('|')
        # if no plural was provided
        except ValueError:
            # make them the same
            singular = plural = label
        # decide which representation of multiplication to use
        op = ' ' if pretty else '*'
        # if the magnitude is exactly one
        if magnitude == one:
            # use the singular form
            return format(magnitude, value) + op + singular
        # otherwise use the plural
        return format(magnitude, value) + op + plural


    # implementation details
    def _strDerivation(self):
        """
        Build a representation of the fundamental unit labels raised to the exponents specified
        in my derivation.

        The unit parser can parse this textual representation and convert it back into a
        dimensional quantity.
        """
        return '*'.join(
            "{}**{}".format(label,exponent) if exponent != 1 else label
            for label, exponent in zip(self.fundamental, self.derivation) if exponent)


# just in case users care about our ordering of the exponents
fundamental = Dimensional.fundamental

# instances
zero = Dimensional(0, Dimensional.zero)
one = dimensionless = Dimensional(1, Dimensional.zero)


# end of file
