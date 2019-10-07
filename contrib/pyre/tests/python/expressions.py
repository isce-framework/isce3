#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise a simple expression calculator
"""

import re

class Calculator(object):

    scanner = re.compile(r"{(?P<identifier>[^}]+)}")

    def compile(self, expression):
        sanitized = self.scanner.sub(self.replace, expression)
        try:
            return compile(sanitized, filename='<expression>', mode='eval')
        except SyntaxError as error:
            raise error

    def eval(self, compiled):
        context = { self.symbolTable[s]: self.valueTable[s] for s in self.symbolTable }
        return eval(compiled, context)

    def replace(self, match):
        identifier = match.group("identifier")

        try:
            return self.symbolTable[identifier]
        except KeyError:
            symbol = "_pyre_op_{0:04d}".format(self.symbols)
            self.symbolTable[identifier] = symbol
            self.symbols += 1
            return symbol

    def __init__(self):
        self.symbols = 0
        self.valueTable = {}
        self.symbolTable = {}
        return


def test():
    e = Calculator()
    e.valueTable = {
        "my value": 4,
        "your value": 2,
        }

    expression = "{my value} * {your value}"
    expression = e.compile(expression)

    assert 8 == e.eval(expression)

    return


# main
if __name__ == "__main__":
    test()


# end of file
