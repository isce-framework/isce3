#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Try to figure out how python decorates, resolves and invokes functions and methods
"""


def test():

    class wrapper(object):

        def overload(self, func):
            # print("wrapper.overload:")
            # print("    name:", func.__name__)

            if func.__name__ != self.default.__name__:
                raise TypeError(
                    "{0.__name__!r} must be called {1.__name__!r}".format(func, self.func))

            flags = func.__code__.co_flags
            if flags & 0x04:
                raise NotImplementedError("*args declarations not supported yet")
            if flags & 0x08:
                raise NotImplementedError("**kwds declarations not supported yet")

            # print("    defaults:", func.__defaults__)
            # print("    kwdefaults:", func.__kwdefaults__)
            # print("    annotations:", func.__annotations__)

            argcount = func.__code__.co_argcount
            signature = func.__code__.co_varnames
            ndefaults = (func.__defaults__ and len(func.__defaults__)) or 0

            positionals = signature[0:(argcount - ndefaults)]
            positionals_default = signature[argcount-ndefaults: argcount]

            kwds = set(signature[argcount:])
            kwdonly_default = (func.__kwdefaults__ and set(func.__kwdefaults__.keys())) or set()
            kwdonly = kwds - kwdonly_default

            # print("  * signature:", signature)
            # print("  * positionals:", positionals)
            # print("  * positionals with default values:", positionals_default)
            # print("  * keywords without default values:", kwdonly)
            # print("  * keywords with default values:", kwdonly_default)

            return self


        def __init__(self, func):
            self.__doc__ = func.__doc__
            self.default = func
            self.signatures = {}

            return


        def __call__(self, *args, **kwds):
            # print("wrapper.__call__:")
            # print("    args:", args)
            # print("    kwds:", kwds)
            return


    def overload(func):
        return wrapper(func)


    @overload
    def sample():
        localvar = 0
        # print("default implementation")
        return

    @sample.overload
    def sample():
        return

    @sample.overload
    def sample(arg0):
        return

    @sample.overload
    def sample(arg0:int):
        return

    @sample.overload
    def sample(arg0=1):
        return

    @sample.overload
    def sample(arg0:int=1):
        return

    @sample.overload
    def sample(arg0, arg1:int=1):
        return

    @sample.overload
    def sample(*, ark0, ark1:int=0, ark2:str="", ark3):
        return

    @sample.overload
    def sample(arg0, arg1:int=1, *, ark0, ark1:int=0, ark2:str="", ark3):
        return

    return


# main
if __name__ == "__main__":
    test()


# end of file
