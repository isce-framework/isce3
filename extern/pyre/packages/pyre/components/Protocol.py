# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import itertools
# support
from .. import tracking
# metaclass
from .Role import Role
# superclass
from .Configurable import Configurable


# class declaration
class Protocol(Configurable, metaclass=Role, internal=True):
    """
    The base class for requirement specifications
    """


    # types
    from ..schemata import uri
    from .exceptions import FrameworkError, DefaultError, ResolutionError
    from .Actor import Actor as actor
    from .Foundry import Foundry as foundry
    from .Component import Component as component


    # framework data
    pyre_key = None
    pyre_isProtocol = True


    # override this in your protocols to provide the default implementation
    @classmethod
    def pyre_default(cls, **kwds):
        """
        The preferred implementation of this protocol, in case the user has not provided an
        alternative
        """
        # actual protocols should override
        return None


    # override this in your protocols to provide custom translations of symbols to component
    # specifications
    @classmethod
    def pyre_convert(cls, value, **kwds):
        """
        Hook to enable protocols to translate the component specification in {value} into a
        canonical form
        """
        # by default, do nothing
        return value


    # introspection
    @classmethod
    def pyre_family(cls):
        """
        Look up my family name
        """
        # get my key
        key = cls.pyre_key
        # if i don't have a key, i don't have a family; otherwise, ask the nameserver
        return cls.pyre_nameserver.getName(cls.pyre_key) if key is not None else None


    @classmethod
    def pyre_familyFragments(cls):
        """
        Look up my family name
        """
        # get my key
        key = cls.pyre_key
        # if i don't have a key, i don't have a family, otherwise, ask the nameserver
        return cls.pyre_nameserver.getSplitName(cls.pyre_key) if key is not None else None


    @classmethod
    def pyre_package(cls):
        """
        Deduce my package name
        """
        # if i don't have a key, i don't have a package
        if cls.pyre_key is None: return None
        # otherwise, get the name server
        ns = cls.pyre_executive.nameserver
        # ask in for the split family name
        family = ns.getSplitName(cls.pyre_key)
        # the package name is the zeroth entry
        pkgName = family[0]
        # use it to look up the package
        return ns[pkgName]


    @classmethod
    def pyre_public(cls):
        """
        Generate the sequence of my public ancestors, i.e. the ones that have a non-trivial family
        """
        # filter public ancestors from my pedigree
        yield from (ancestor for ancestor in cls.pyre_pedigree if ancestor.pyre_key is not None)
        # all done
        return


    # support for framework requests
    @classmethod
    def pyre_resolveSpecification(cls, spec, **kwds):
        """
        Attempt to resolve {spec} into a component that implements me; {spec} is
        assumed to be a string
        """
        # get the executive
        executive = cls.pyre_executive
        # and ask it to resolve {value} into component candidates; this will handle correctly
        # uris that resolve to a retrievable component, as well as uris that mention an
        # existing instance
        for candidate in executive.resolve(uri=spec, protocol=cls, **kwds):
            # we have a candidate; now it's time to decide whether it is acceptable. the
            # definition of what's acceptable is somewhat complicated because there are many
            # sensible use cases to support. the first step is to check whether the candidate
            # fulfills the obligations defined my this protocol, i.e. it has the correct
            # properties and behaviors. then, there are issues of pedigree that must be resoved
            # on a case by case basis

            # if it is compatible with my protocol
            if candidate.pyre_isCompatible(spec=cls):
                # show me
                # print('pyre.components.Protocol: compatible candidate: {}'.format(candidate))
                # we are done
                return candidate

        # if we get this far, i just couldn't pull it off
        raise cls.ResolutionError(protocol=cls, value=spec)


    @classmethod
    def pyre_resolutionContext(cls):
        """
        Return an iterable over portions of my family name
        """
        # go through my public ancestors, get the family name fragments and send them off
        yield from ( base.pyre_familyFragments() for base in cls.pyre_public() )
        # all done
        return


    @classmethod
    def pyre_locateAllImplementers(cls):
        """
        Retrieve all visible components that are compatible with me

        For components to be visible, their location has to be deducible based only on the
        information available to this protocol; other compatible components may be provided by
        packages that have not been imported yet, or live in files outside the canonical layout
        """
        # all registered implementers
        yield from cls.pyre_locateAllRegisteredImplementers()
        # all loadable implementers
        yield from cls.pyre_locateAllLoadableImplementers()
        # all importable implementers
        yield from cls.pyre_locateAllImportableImplementers()
        # all done
        return


    @classmethod
    def pyre_locateAllRegisteredImplementers(cls):
        """
        Retrieve all implementers that are already registered with the framework

        This catches components whose source has been seen by the framework at the time this
        method was invoked; the information available may be different during a subsequent call
        if pyre has bumped into additional implementers
        """
        # get the registrar
        registrar = cls.pyre_registrar
        # traverse my inheritance
        for ancestor in cls.pyre_public():
            # access the registered implementers for this ancestor
            for implementer in registrar.implementers[ancestor]:
                # if this is not a publicly visible entity
                if not implementer.pyre_isPublicClass():
                    # skip it
                    continue
                # get the family name; this is equivalent to the fully scoped {uri}
                uri = implementer.pyre_family()
                # extract the short name
                name = implementer.pyre_familyFragments()[-1]
                # and add it to the pile
                yield uri, name, implementer
        # all done
        return

    @classmethod
    def pyre_locateAllImportableImplementers(cls):
        """
        Retrieve all implementers registered in a namespace derivable from my family name
        """
        # splice my family name together to form a module name
        uri = '.'.join(cls.pyre_familyFragments())
        # if i don't have a public name, there is nothing to do
        if not uri: return
        # attempt to
        try:
            # hunt the implementers down
            yield from cls.pyre_implementers(uri='import:{}'.format(uri))
        # if anything goes wrong
        except cls.FrameworkError:
            # skip this step
            pass
        # all done
        return


    @classmethod
    def pyre_locateAllLoadableImplementers(cls):
        """
        Retrieve all implementers that live in files and folders derivable from my family name
        """
        # get my family fragments
        fragments = cls.pyre_familyFragments()
        # if i don't have a public name, there is nothing to do
        if not fragments: return
        # get the file server
        vfs = cls.pyre_fileserver
        # construct the base uri
        uri = vfs.path(*fragments)
        # try to
        try:
            # get the associated node
            node = vfs[uri]
        # if it's not there
        except vfs.NotFoundError:
            # no worries
            pass
        # if it is there
        else:
            # reset the workload
            todo = [ (uri, node) ]
            # go through all nodes
            for path, folder in todo:
                # grab the contents
                for name, child in folder.open():
                    # form the path to the child
                    name = path / name
                    # if the child is a folder
                    if child.isFolder:
                        # put it on the to do pile
                        todo.append((name, child))
                    # otherwise
                    else:
                        # treat it as a shelf; assemble its address
                        shelf = 'vfs:{}'.format(name)
                        # and get its contents
                        yield from cls.pyre_implementers(uri=shelf)

        # the last thing to try is a shelf named after my family
        uri = uri.withSuffix(suffix='.py')
        # check whether
        try:
            # such a node exists
            node = vfs[uri]
        # if not
        except vfs.NotFoundError:
            # no worries
            pass
        # otherwise
        else:
            # yield its contents
            yield from cls.pyre_implementers(uri='vfs:'+str(uri))

        # all done
        return


    @classmethod
    def pyre_implementers(cls, uri):
        """
        Retrieve components that are compatible with me from the shelf in {uri}
        """
        # get the shelf contents
        for name, entity in cls.pyre_executive.retrieveComponents(uri=uri):
            # if the entity is a component
            if isinstance(entity, cls.actor):
                # and it is compatible with me
                if entity.pyre_isCompatible(spec=cls, fast=True):
                    # pass it along
                    yield uri, name, entity
                # grab the next one
                continue

            # if the entity is a foundry
            if isinstance(entity, cls.foundry):
                # check whether any of its protocols
                for protocol in entity.pyre_implements:
                    # is compatible with me
                    if protocol.pyre_isCompatible(spec=cls, fast=True):
                        # in which case, pass it along
                        yield uri, name, entity
                        # stop checking other protocols
                        break
                # grab the next one
                continue

            # other kinds?
            import journal
            return journal.firewall.log("new kind of symbol in shelf {!r}".format(str(uri)))

        # all done
        return


    # compatibility checks
    @classmethod
    def pyre_isCompatible(cls, spec, fast=True):
        """
        Check whether {this} protocol is compatible with the {other}
        """
        # print("PP: me={}, other={}".format(cls, spec))
        # first, the easy cases am i looking in the mirror?
        if cls == spec:
            # easy; no need to build a report since the rest of the code is not supposed to
            # touch the report unless it evaluates to {False}
            return True

        # i am never compatible with components...
        if spec.pyre_isComponent:
            # in fact, let's treat asking the question as a bug
            import journal
            # so complain
            raise journal.firewall('pyre.components').log(
                'PC compatibility checks are not supported')

        # do the assignment compatibility check
        report = super().pyre_isCompatible(spec=spec, fast=fast)
        # if we are in fast mode and got an error
        if fast and report:
            # all done
            return report

        # all done
        return report


    @classmethod
    def pyre_isTypeCompatible(cls, protocol):
        """
        Decide whether {cls} is type compatible with the given {protocol}

        The intent here is to enable protocols to tighten or loosen this definition to fit
        their specific use cases. The implementation is based on pedigree comparisons between
        {cls} and {protocol}
        """
        # show me
        # print("me: {}, other: {}".format(cls, protocol))

        # I am automatically compatible with my ancestors
        if issubclass(cls, protocol):
            # print('  ** subclass **')
            # we are compatible
            return True

        # it is possible for protocol to be an ancestor, yet the subclass check fails; this
        # happens when protocols and components are dynamically imported from the filesystem as
        # part of the specification resolution process; we need an additional check in this case

        # the following works well enough for protocols with families; we can use the hash key
        # to identify ancestor classes

        # grab the hash key of the target
        key = protocol.pyre_key
        # if it's a public protocol
        if key:
            # go through each of my ancestors
            for ancestor in cls.pyre_pedigree:
                # get my key
                mine = ancestor.pyre_key
                # if they are the same
                if mine == key:
                    # we have a match
                    return True

        # in order to support downward compatibility by packages that subclass protocols for
        # administrative reasons without adding any new requirements, we permit the reverse of
        # the above; it may sound strange, but it should be safe. we have already done the
        # trait check, so we can relax the type check a little bit

        # case in point are actions, which currently do some framework magic to ensure that
        # actions inherited from ancestral packages are retrievable. we may drop this
        # stretching here as soon as that's implemented in a better way

        # we can't stretch too far though; we should not allow {cls} and {protocol} to be
        # compatible if they have a common ancestor; you don't want a blas instance to satisfy
        # the hdf5 requirement, just because they are both libraries...

        # grab my hash key
        mine = cls.pyre_key
        # if i am a public protocol
        if mine:
            # go through each of the target's ancestors
            for ancestor in protocol.pyre_pedigree:
                # get her key
                hers = ancestor.pyre_key
                # if they are the same
                if mine == hers:
                    # we have a match
                    return True

        # otherwise
        return False


    # constants
    EXTENSION = '.py'


# end of file
