# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import sys
# support
from ... import primitives, tracking
# superclass
from ..Loader import Loader


# declaration
class Importer(Loader):
    """
    This component codec recognizes uris of the form

        import:package.subpackage.factory#name

    The uri is interpreted as if

        from package.subpackage import factory
        factory(name=name)

    had been issued to the interpreter. {factory} is expected to be either a component class or
    a function that returns a component class. This class is then instantiated using {name} as
    the sole argument to the constructor. If {name} is not present, the component class is
    returned.
    """


    # types
    from .Shelf import Shelf as shelf


    # public data
    schemes = ('import',)


    # interface
    @classmethod
    def load(cls, uri, **kwds):
        """
        Interpret {uri} as a module to be loaded
        """
        # get the module name
        source = str(uri.address)
        # build a simple locator
        locator = tracking.simple(source=uri.uri)
        # show me
        # print("    importing: {!r}".format(source))
        # attempt to
        try:
            # import the module
            module = __import__(source)
        # the address portion of {uri} is not importable
        except (ImportError, TypeError) as error:
            # show me
            # print("      error: {}".format(str(error)))
            # complain
            raise cls.LoadingError(
                codec=cls, uri=uri, locator=locator, description=str(error)) from error
        # all other exceptions are probably caused by the contents of the module; let them
        # propagate to the user; on success, look up {module} in the global list of modules and
        # return it dressed up as a shelf
        return cls.shelf(module=sys.modules[source], uri=uri, locator=locator)


    @classmethod
    def locateShelves(cls, protocol, scheme, context, **kwds):
        """
        Locate candidate shelves for the given {uri}
        """
        # sign in
        # print("{.__name__}.locateShelves:".format(cls))

        # chain up for the rest
        for candidate in super().locateShelves(
                protocol=protocol, scheme=scheme, context=context, **kwds):
            # make a uri
            uri = cls.uri(scheme='import', address=candidate)
            # and send it off
            yield uri

        # all done
        return


    # context handling
    @classmethod
    def interpret(cls, request):
        """
        Attempt to extract to extract a resolution context and a symbol from the {request}
        """
        # i deal with python package specifications
        context = request.address.split('.')
        # the symbol is just the last entry
        symbol = '' if not context else context[-1]
        # return the pair
        return context, symbol


    @classmethod
    def assemble(cls, context):
        """
        Assemble the sequence of directories in to a form suitable for the address part of a uri
        """
        # i make module paths
        return '.'.join(context)


    # initialization
    @classmethod
    def prime(cls, linker):
        """
        Build my initial set of shelves
        """
        # attempt to
        try:
            # get the main module
            import __main__
        # if this failed
        except ImportError:
            # no worries
            return

        # otherwise, attempt to
        try:
            # get the name of the script we are executing
            filename = __main__.__file__
        # if it doesn't have one
        except AttributeError:
            # no worries
            return

        # resolve the file name
        filename = str(primitives.path(filename).resolve())
        # make a uri
        uri = cls.uri(scheme='file', address=filename)
        # and a locator
        here = tracking.simple('while priming the {.__name__} loader'.format(cls))
        # make a shelf
        shelf = cls.shelf(module=__main__, uri=uri, locator=here)
        # attach it to the linker
        linker.shelves[uri.uri] = shelf
        # show me
        # print("registered '__main__' as {.uri!r}".format(uri))

        # nothing else to do
        return


# end of file
