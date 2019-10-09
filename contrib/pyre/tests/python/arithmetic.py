#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
A simple expression scanner that understands some whitespace as mutliplication
"""

import re

class Simplifier(object):

    scanner = re.compile(
        r"(?P<surrounding>\s*(?P<operator>[*/+-]{1})\s*)"
        r"|"
        r"(?P<open>\s*(?P<oparen>[(]+)\s*)"
        r"|"
        r"(?P<close>\s*(?P<cparen>[)]+)\s*)"
        r"|"
        r"(?P<whitespace>[\s]+)"
        )

    def simplify(self, expression):
        return "*".join(self.scanner.sub(self.replace, expression).split())

    def replace(self, match):
        if match.group("whitespace"):
            return " "
        if match.group("open"):
            return " " + match.group("oparen")
        if match.group("close"):
            return match.group("cparen") + " "
        if match.group("surrounding"):
            return match.group("operator")
        return ""

    def __init__(self):
        self.symbols = 0
        self.valueTable = {}
        self.symbolTable = {}
        return


def test():
    s = Simplifier()

    expr = "12      a      (((b          +     c  )d))           d/         4"
    assert s.simplify(expr) == "12*a*(((b+c)*d))*d/4"

    return


# main
if __name__ == "__main__":
    test()


# end of file
