#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
A comprehensive test of arithmetic operator overloading
"""

class Node:
    """A sample object that supports algebraic expressions among its instances and floats"""

    # public data
    value = None

    # meta methods
    def __init__(self, value):
        self.value = value

    # algebra
    def __add__(self, other):
        if isinstance(other, Node):
            value = self.value + other.value
        else:
            value = self.value + other
        return type(self)(value=value)

    def __sub__(self, other):
        if isinstance(other, Node):
            value = self.value - other.value
        else:
            value = self.value - other
        return type(self)(value=value)

    def __mul__(self, other):
        if isinstance(other, Node):
            value = self.value * other.value
        else:
            value = self.value * other
        return type(self)(value=value)

    def __truediv__(self, other):
        if isinstance(other, Node):
            value = self.value / other.value
        else:
            value = self.value / other
        return type(self)(value=value)

    def __floordiv__(self, other):
        if isinstance(other, Node):
            value = self.value // other.value
        else:
            value = self.value // other
        return type(self)(value=value)

    def __mod__(self, other):
        if isinstance(other, Node):
            value = self.value % other.value
        else:
            value = self.value % other
        return type(self)(value=value)

    def __divmod__(self, other):
        if isinstance(other, Node):
            d, m = divmod(self.value, other.value)
        else:
            d, m = divmod(self.value, other)
        return type(self)(value=d), type(self)(value=m)

    def __pow__(self, other):
        if isinstance(other, Node):
            value = self.value ** other.value
        else:
            value = self.value ** other
        return type(self)(value=value)

    def __radd__(self, other):
        value = self.value + other
        return type(self)(value=value)

    def __rsub__(self, other):
        value = other - self.value
        return type(self)(value=value)

    def __rmul__(self, other):
        value = self.value * other
        return type(self)(value=value)

    def __rtruediv__(self, other):
        value = other / self.value
        return type(self)(value=value)

    def __rfloordiv__(self, other):
        value = other // self.value
        return type(self)(value=value)

    def __rmod__(self, other):
        value = other % self.value
        return type(self)(value=value)

    def __rdivmod__(self, other):
        d, m = divmod(other, self.value)
        return type(self)(value=d), type(self)(value=m)

    def __rpow__(self, other):
        value = other ** self.value
        return type(self)(value=value)

    def __iadd__(self, other):
        self.value += other
        return self

    def __isub__(self, other):
        self.value -= other
        return self

    def __imul__(self, other):
        self.value *= other
        return self

    def __itruediv__(self, other):
        self.value /= other
        return self

    def __ifloordiv__(self, other):
        self.value //= other
        return self

    def __imod__(self, other):
        self.value %= other
        return self

    def __ipow__(self, other):
        self.value **= other
        return self

    def __neg__(self):
        return type(self)(value=-self.value)

    def __pos__(self):
        return self

    def __abs__(self):
        return type(self)(value=abs(self.value))


def test():
    # declare a couple of nodes
    n1 = Node(value=1)
    n2 = Node(value=2)
    # unary operators
    assert (- n1).value == -1
    assert (+ n2).value == 2
    assert (abs(n1)).value == 1
    # basic arithmetic with two operands
    assert (n1 + n2).value == 1 + 2
    assert (n1 - n2).value == 1 - 2
    assert (n1 * n2).value == 1 * 2
    assert (n1 / n2).value == 1 / 2
    assert (n1 // n2).value == 1 // 2
    # basic arithmetic with more than two operands
    assert (n1 + n2 - n1).value == 1 + 2 - 1
    assert (n1 * n2 / n1).value == 1 * 2 / 1
    assert ((n1 - n2)*n2).value == (1 - 2)*2
    # basic arithmetic with floats
    assert (1 + n2).value == 1 + 2
    assert (n2 + 1).value == 2 + 1
    assert (1 - n2).value == 1 - 2
    assert (n2 - 1).value == 2 - 1
    assert (2 * n1).value == 2 * 1
    assert (n1 * 2).value == 1 * 2
    assert (3 / n2).value == 3 / 2
    assert (n2 / 3).value == 2 / 3
    assert (n2 ** 3).value == 2**3
    assert (3 ** n2).value == 3**2

    # more complicated forms
    assert ((n1**2 + 2*n1*n2 + n2**2)).value == ((n1+n2)**2).value
    assert ((n1**2 - 2*n1*n2 + n2**2)).value == ((n1-n2)**2).value
    assert (2*(.5 - n1*n2 + n2**2)*n1).value == 2*(.5 - 1*2 + 2**2)*1

    return


# main
if __name__ == "__main__":
    test()


# end of file
