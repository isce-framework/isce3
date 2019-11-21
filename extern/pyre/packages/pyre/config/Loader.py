# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import itertools


# declaration
class Loader:
    """
    Base class for strategies that build component descriptors from persistent stores
    """


    # types
    from ..primitives import uri
    from .exceptions import LoadingError


    @classmethod
    def loadShelves(cls, executive, protocol, uri, scheme, context, **kwds):
        """
        Locate and load shelves for the given {uri}; if the {uri} is not sufficiently qualified
        to point to a unique location, use {protocol} to form plausible candidates.
        """
        # sign in
        # print("{.__name__}.loadShelves:".format(cls))
        # print("    protocol: {}".format(protocol))
        # print("    scheme: {}".format(scheme))
        # print("    context: {}".format(context))
        # for key, value in kwds.items():
            # print("    {}: {}".format(key, value))

        # access the linker
        linker = executive.linker
        # print(" -- priming the search for shelves")
        # use {protocol} to build a sequence of candidate locations
        candidates = cls.locateShelves(executive=executive, protocol=protocol,
                                       scheme=scheme, context=context,
                                       **kwds)
        # print(" -- done priming the search for shelves")
        # go through each of them
        for candidate in candidates:
            # show me
            # print(" -- trying shelf={.uri!r}".format(candidate))
            # does this uri correspond to a known shelf
            try:
                # if yes, grab it
                shelf = linker.shelves[candidate.uri]
                # show me
                # print("    shelf {!r} previously loaded".format(candidate.uri))
            # otherwise
            except KeyError:
                # show me
                # print("    new shelf; loading")
                # make an empty shelf and register it with the
                # linker to prevent it from attempting to load this shelf again, in case there
                # are loading side effects
                linker.shelves[candidate.uri] = cls.shelf(uri=candidate)
                # attempt to
                try:
                    # load it
                    shelf = cls.load(executive=executive, uri=candidate)
                # if it fails
                except cls.LoadingError as error:
                    # show me
                    # print(" ## skipping: {}".format(error))
                    # remove the bogus registration
                    del linker.shelves[candidate.uri]
                    # move on to the next candidate
                    continue
                # if the shelf was loaded correctly, replace the bogus registration
                linker.shelves[candidate.uri] = shelf
                # show me
                # print("      success; registering {!r} with the linker".format(candidate.uri))

            # yield the shelf to my caller
            yield shelf

        # no more candidates
        return


    @classmethod
    def locateShelves(cls, executive, protocol, scheme, context, symbol, cfgpath=None, **kwds):
        """
        Locate candidate shelves for the given {uri}
        """
        # sign in
        # print("{.__name__}.locateShelves:".format(cls))

        # first, let's exhaust the user's specification; we build a tuple with progressively
        # shorter non-empty leading subsequences of the context. note that it includes the
        # entity that we interpret as the {symbol}: see the comments at the bottom of this
        # method
        contexts = list(
            # the subsequence
            context[:pos]
            # the indices
            for pos in reversed(range(1, len(context)+1)))
        # show me
        # print("    contexts: {}".format(contexts))

        # we will need this later; for now:
        for candidate in contexts:
            # send it for evaluation
            yield cls.assemble(candidate)

        # if we get this far, the user's specification was not enough. there are two more
        # sources of information available. first, there may be a registered app that can
        # provide resolution context
        app = executive.dashboard.pyre_application
        # if we have a non-trivial app, prime the {prefixes} with its searchpath; the search
        # path is the set of packages that contain its public ancestors
        prefixes = list(app.searchpath) if app else []

        # protocols contribute to the resolution context through their family names. the
        # leading part is a package, which we add to the list of prefixes; the remainder of the
        # protocol is its flavor. we use the flavor to effect a kind of inheritance:
        # applications form the package {foo} can have their own actions, but also inherit the
        # standard {pyre} actions. this happens automatically if the protocol {pyre.actions} is
        # encountered while instantiating a {foo} app
        flavors = []
        # if we have a protocol
        if protocol:
            # go through the protocol resolution context and unpack the package name and the
            # protocol flavor
            for package, *flavor in protocol.pyre_resolutionContext():
                # if we got a new package
                if package not in prefixes:
                    # save it
                    prefixes.append(package)
                # go through the flavors we've seen before and verify that the new flavor is
                # not a subsequence of any of the others
                for known in flavors:
                    # pairwise check
                    for new, old in zip(flavor, known):
                        # for equality
                        if new != old: break
                    # if we didn't find any discrepancy
                    else:
                        # it means that we exhausted one of them with no mismatch; if the known
                        # is longer
                        if len(known) >= len(flavor):
                            # ignore this one
                            break
                # if the search terminated normally
                else:
                    # add this flavor to the pile
                    flavors.append(flavor)

        # add any extra locations supplied by the subclass loader
        cfgpath = [[]] + (list([path] for path in cfgpath) if cfgpath else [])
        # convert {prefixes} into a list of lists
        prefixes = [[]] + (list([prefix] for prefix in prefixes) if prefixes else [])
        # repeat with the flavors
        flavors = flavors if flavors else [[]]

        # show me
        # print("  prefixes: {}".format(prefixes))
        # print("  flavors: {}".format(flavors))
        # print("  context: {}".format(contexts))

        # keep track of what we have tried
        candidates = []
        # here is the list of product arguments
        combine = (cfgpath, prefixes, flavors, contexts)
        # form all possible combinations of (prefix, flavor, context)
        for path, prefix, flavor, user in itertools.product(*combine):
            # show me
            # print(' -- path: {}'.format(path))
            # print(' -- prefix: {}'.format(prefix))
            # print(' -- flavor: {}'.format(flavor))
            # print(' -- user: {}'.format(user))
            # now, slide a splicer through all positions in the flavor past the first slot
            for pos in reversed(range(len(flavor)+1)):
                # show me
                # print(' ++ pos: {}'.format(pos))
                # keep the front part
                front = flavor[:pos]
                # we use it to form a candidate
                candidate = cls.assemble(path + prefix + front + user)
                # show me
                # print("    candidate: {}".format(candidate))
                # and send it for inspection
                yield candidate
                # remember this one
                candidates.append(candidate)
                # now for each flavor level we spliced off
                for spliced in reversed(flavor[pos:]):
                    # use them to form candidates as well
                    candidate = cls.assemble(path + prefix + front + [spliced])
                    # show me
                    # print("    candidate: {}".format(candidate))
                    # and send it for inspection
                    yield candidate
                    # remember this one too
                    candidates.append(candidate)

        # MGA@20160415: the following trick doesn't seem to be necessary anymore, as it appears
        # that the interpretation of the symbol as a shelf is now attempted during the normal
        # expansion. i commented the section out, rather than removing it, just in case there
        # is some corner case that i haven't considered here

        # now, for my last trick: attempt to interpret the symbol itself as a shelf
        # print(" ++ interpreting {!r} as a shelf".format(symbol))
        # send it off
        # yield symbol

        # all done
        return


    # initialization
    @classmethod
    def register(cls, index):
        """
        Register the recognized schemes with {index}
        """
        # update the index with my schemes
        index.update((scheme, cls) for scheme in cls.schemes)
        # all done
        return


    @classmethod
    def prime(cls, linker):
        """
        Build my initial set of shelves
        """
        # nothing to do
        return


# end of file
