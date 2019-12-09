# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access to the locator factories
from ... import primitives, tracking
# and my ancestors
from ..Loader import Loader


class ODB(Loader):
    """
    This component codec recognizes uris of the form

        vfs:/path/module.py/factory#name
        file:/path/module.py/factory#name

    which is interpreted as a request to import the file {module.py} from the indicated path,
    look for the symbol {factory}, and optionally instantiate whatever component class is
    recovered using {name}
    """


    # type
    from .Shelf import Shelf as shelf


    # constants
    suffix = '.py'
    schemes = ('vfs', 'file')


    # interface
    @classmethod
    def load(cls, executive, uri, **kwds):
        """
        Interpret {uri} as a shelf to be loaded
        """
        # get the fileserver
        fs = executive.fileserver
        # show me
        # print("    loading: {.uri!r}".format(uri))
        # ask it to
        try:
            # open the {uri}
            stream = fs.open(uri=uri)
        # if anything goes wrong
        except fs.GenericError as error:
            # show me
            # print("      error: {}".format(str(error)))
            # report it as a loading error
            raise cls.LoadingError(codec=cls, uri=uri) from error
        # build a new shelf
        shelf = cls.shelf(stream=stream, uri=uri, locator=tracking.file(source=str(uri)))
        # and return it
        return shelf


    @classmethod
    def locateShelves(cls, executive, protocol, scheme, context, **kwds):
        """
        Locate candidate shelves from the given {uri}
        """
        # sign in
        # print("{.__name__}.locateShelves:".format(cls))

        # if there is no scheme
        if not scheme:
            # set it to the defaults
            scheme = 'vfs'
        # first, let's try what the user supplied
        # uri = cls.uri(scheme=scheme, address=cls.assemble(context))
        # show me
        # print(" ++ trying the user's uri={.uri!r}".format(uri))
        # and try it
        # yield uri

        # collect the list of system folders maintained by the fileserver
        cfgpath = list(str(folder) for folder in executive.fileserver.systemFolders)

        # chain up for the rest
        for candidate in super().locateShelves(
                executive=executive, cfgpath=cfgpath,
                protocol=protocol, scheme=scheme, context=context, **kwds):
            # make a uri
            uri = cls.uri(scheme=scheme, address=candidate)
            # show me
            # print("  candidate uri={.uri!r}".format(uri))
            # and send it off
            yield uri

        # all done
        return


    # context handling
    @classmethod
    def interpret(cls, request):
        """
        Attempt to extract a resolution context and a symbol from the {request}
        """
        # i deal with paths, so attempt to coerce the request
        path = primitives.path(request.address)
        # the resolution context is the tuple of directories in the {request}, ignoring the
        # root marker if any
        context = list(path.names)
        # and the symbol is just the path name with any suffixes
        symbol = path.stem
        # return the pair; perhaps i can skip the realization and hand back the generator...
        return context, symbol


    @classmethod
    def assemble(cls, context):
        """
        Assemble the sequence of directories in {context} to a form suitable for the address part
        of a uri
        """
        # i make paths
        path = primitives.path(context)
        # if it has no suffix
        if not path.suffix:
            # give it one...
            path = path.withSuffix(cls.suffix)
        # turn it into a string and return it
        return str(path)


# end of file
