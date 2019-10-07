# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import collections


# declaration
class Linker:
    """
    Class responsible for accessing components from a variety of persistent stores
    """


    # types
    from .exceptions import FrameworkError, ComponentNotFoundError, BadResourceLocatorError
    from ..schemata import uri


    # public data
    codecs = None
    shelves = None


    # support for framework requests
    def loadShelf(self, uri, **kwds):
        """
        Load the shelf specified by {uri}
        """
        # coerce the uri
        uri = self.uri().coerce(uri)
        # get the codec
        codec = self.schemes[uri.scheme]
        # ask it to load the shelf and return it
        return codec.load(uri=uri, **kwds)


    def resolve(self, executive, uri, protocol, **kwds):
        """
        Attempt to locate the component class specified by {uri}
        """
        # get the scheme
        scheme = uri.scheme

        # if the {uri} has a scheme, find the associated codec and get it to interpret the
        # resolution request. if the {uri} has no scheme, hand the request to each codec in the
        # order they were registered
        try:
            # make the pile
            codecs = [ self.schemes[scheme] ] if scheme else self.codecs
        # and if the look up fails
        except KeyError:
            # it's because we don't recognize the scheme
            reason = "unknown scheme {!r}".format(scheme)
            # so complain
            raise self.BadResourceLocatorError(uri=uri, reason=reason)

        # each codec will interpret the uri and provide a hint as to what the user is looking
        # for; let's remember these attempts so we can try some non-obvious things when all
        # else fails
        symbols = set()

        # go through the relevant codecs
        for codec in codecs:
            # attempt to interpret the address as specifying a symbol from a candidate
            # container. each codec has its own way of doing that. the wrong ones produce
            # nonsense, so we have to be careful...
            context, symbol = codec.interpret(request=uri)
            # add the symbol to the pile
            symbols.add(symbol)

            # and ask each one for all relevant shelves
            for shelf in codec.loadShelves(executive=executive, protocol=protocol, uri=uri,
                                           scheme=scheme, context=context, symbol=symbol,
                                           **kwds):
                # got one; show me
                # print('    shelf contents: {}'.format(shelf))
                # check whether it contains our symbol by attempting to
                try:
                    # extract it
                    descriptor = shelf.retrieveSymbol(symbol)
                    # if it's not there
                except shelf.SymbolNotFoundError as error:
                    # show me
                    # print(' ## error: {}'.format(str(error)))
                    # not there; try the next match
                    continue
                # otherwise, we have a candidate; show me
                # print(' ## got: {}'.format(descriptor))
                # let the client evaluate it further
                yield descriptor

        # ok, no dice. can we get some help from the protocol?
        if not protocol:
            # not there; giving up
            return

        # we have exhausted all supported cases of looking at external sources; there is one
        # more thing to try: in the process of interpreting the user request, we formed guesses
        # regarding the name the user is looking for. perhaps there is an implementer of our
        # protocol whose package name is the symbol we are looking for

        # look through the protocol implementers
        for implementer in executive.registrar.implementers[protocol]:
            # get their package names
            package = implementer.pyre_package()
            # and yield ones whose package name matches our symbol candidates
            if package and package.name in symbols:
                # let the user evaluate further
                yield implementer

        # out of ideas
        return


    # meta-methods
    def __init__(self, executive, **kwds):
        # chain up
        super().__init__(**kwds)

        # the map from uris to known shelves
        self.shelves = {}
        # setup my default codecs and initialize my scheme index
        codecs, schemes = self.indexDefaultCodecs()
        # save them
        self.codecs = codecs
        self.schemes = schemes

        # go through the set of registered codecs
        for codec in codecs:
            # and prime each one
            codec.prime(linker=self)

        # nothing else
        return


    # implementation details
    def indexDefaultCodecs(self):
        """
        Initialize my codec index
        """
        # get the codecs i know about
        from ..config.odb import odb
        from ..config.native import native
        # put them in a pile
        codecs = [odb, native]

        # make an empty index
        schemes = collections.OrderedDict()
        # register the native codec
        native.register(index=schemes)
        # register the file loader
        odb.register(index=schemes)

        # all done
        return codecs, schemes


# end of file
