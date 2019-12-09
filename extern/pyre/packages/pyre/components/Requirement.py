# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
from .. import tracking
# superclass
from ..patterns.AttributeClassifier import AttributeClassifier


# class declaration
class Requirement(AttributeClassifier):
    """
    Metaclass that enables the harvesting of trait declarations

    This class captures the class record processing that is common to both interfaces and
    components. Given a declaration record, {Requirement}

      * discovers the bases classes that are configurables
      * identifies the specially marked attributes
      * creates the namemap that handles trait name aliasing
    """
    # the presence of a family/component name determines the slot storage strategy: if a
    # configurable is to be registered with the nameserver, it delegates storage of its slot to
    # it, which allows me to maintain only one slot for each component trait. if a configurable
    # is not registered, its slots are kept in its pyre_inventory. slotted traits know how to
    # retrieve the slots for each kind

    # at one point, i tossed around the idea of introducing accessors to handle the different
    # strategies for storing slots. it turned out to be very tricky to do it that way: each
    # class in the inheritance graph had to get its own accessors for all its slots, both local
    # and inherited, which made support for shadowing very tricky.

    # types
    from ..traits.Trait import Trait
    from .Configurable import Configurable


    # meta-methods
    def __new__(cls, name, bases, attributes, *, internal=False, **kwds):
        """
        Build the class record for a new configurable

        This metaclass handles the record building steps common to components and protocols,
        whose metaclasses are subclasses of {Requirement}. It initializes the class record so
        that it matches the layout of {Configurable}, the base class of {Component} and
        {Protocol}.
        """
        # my local traits
        # access to my name maps
        namemap = {}
        traitmap = {}

        # initialize the class attributes explicitly
        attributes['pyre_pedigree'] = ()
        attributes["pyre_localTraits"] = ()
        attributes['pyre_inheritedTraits'] = ()
        attributes['pyre_namemap'] = namemap
        attributes['pyre_traitmap'] = traitmap
        attributes['pyre_internal'] = internal

        # build the record
        configurable = super().__new__(cls, name, bases, attributes, **kwds)

        # harvest the local traits
        configurable.pyre_localTraits = tuple(
            configurable.pyre_getLocalTraits(attributes))

        # harvest the inherited traits: this must be done from scratch for every new
        # configurable class, since multiple inheritance messes with the __mro__;
        configurable.pyre_inheritedTraits = tuple(
            configurable.pyre_getInheritedTraits(shadowed=set(attributes)))

        # extract the ancestors listed in the mro of the configurable that are themselves
        # configurable; N.B.: since {Requirement} is not the direct metaclass of any class, the
        # chain here stops at either {Component} or {Protocol}, depending on whether {Actor}
        # or {Role} is the actual metaclass
        configurable.pyre_pedigree = tuple(configurable.pyre_getPedigree())

        # adjust the name maps; the local variables are tied to the class attribute
        # N.B. this assumes that the traits have been initialized, which updates the {aliases}
        # to include the canonical name of the trait
        for trait in configurable.pyre_traits():
            # update the trait map
            traitmap[trait.name] = trait
            # update the namemap with all aliases of each trait
            namemap.update({alias: trait.name for alias in trait.aliases})

        # return the class record to the caller
        return configurable


    # support for decorating components and protocols
    def pyre_getLocalTraits(self, attributes):
        """
        Scan the dictionary {attributes} for trait descriptors
        """
        # examine the attributes and harvest the trait descriptors
        for name, trait in self.pyre_harvest(attributes=attributes, descriptor=self.Trait):
            # establish my association with my trait
            trait.bind(client=self, name=name)
            # add it to the pile
            yield trait
        # all done
        return


    def pyre_getInheritedTraits(self, shadowed):
        """
        Look through the ancestors of {configurable} for traits whose name are not members of
        {shadowed}, the set of names that are inaccessible.
        """
        # N.B.: this used to filter ancestors based simply on whether they were instances of my
        # metaclass. See the note at {pyre_getPedigree} below for reasons why this was not a
        # good solution

        # print("{.__name__!r}: harvesting inherited traits".format(self))
        # for each superclass of configurable
        for base in self.__mro__[1:]:
            # print("    looking through {.__name__!r}".format(base))
            # try to
            try:
                # get its local traits
                traits = base.pyre_localTraits
            # if this fails
            except AttributeError:
                # not a problem
                # print("        no traits")
                pass
            # if it succeeds
            else:
                # bail out if we have reached the end of the configurable chain
                if base is self.Configurable: return
                # go through the traits local to this base
                for trait in traits:
                    # print("        found {!r}".format(trait.name))
                    # skip it if something else by the same name is already known
                    if trait.name in shadowed: continue
                    # otherwise, save it
                    yield trait
            # in any case, add all the local attribute names onto the known pile
            shadowed.update(base.__dict__)
            # print("    done")

        # all done
        return


    def pyre_getPedigree(self):
        """
        Visit my ancestors and locate the ones that are themselves configurables
        """
        # N.B.: this used to be implemented as a simple check of whether a given {base} was an
        # instance of my metaclass. it turns out that this algorithm fails for subclasses of
        # configurables that have their own metaclass, such as {pyre.shells.Application} whose
        # metaclass {pyre.shells.Director} fails to recognize {Component} subclasses as its
        # instances. the net effect was that any trait defined in an {Application} ancestor
        # component would be ignored during the computation of inherited traits, making certain
        # factorizations impossible

        # for each class in the chain
        for base in self.__mro__:
            # check whether
            try:
                # the base has a {pyre_localTraits} attribute
                base.pyre_localTraits
            # if not
            except AttributeError:
                # no problem; move on
                continue
            # otherwise
            else:
                # one of ours; if it is the end of the chain, stop looking
                if base is self.Configurable: return
                # otherwise, hand it to our caller
                yield base

        # all done
        return


    def pyre_getEigenPedigree(self):
        """
        Build a sequence of my ancestors that are configurable and NOT related to each other
        through inheritance
        """
        # make a pile of known ancestors
        known = set()
        # loop over my pedigree
        for base in self.pyre_pedigree:
            # ignore the trivial and the known
            if base.pyre_internal or base in known: continue
            # got one
            yield base
            # put its pedigree in the known pile
            known.update(base.pyre_pedigree)
        # all done
        return


    # type checks
    @classmethod
    def pyre_isComponent(self, configurable):
        """
        Check whether {configurable} is a component
        """
        # try to
        try:
            # ask it
            flag = configurable.pyre_isComponent
        # if it doesn't know how to answer the question
        except AttributeError:
            # it isn't
            flag = False
        # all done
        return flag


    @classmethod
    def pyre_isProtocol(self, configurable):
        """
        Check whether {configurable} is a protocol
        """
        # try to
        try:
            # ask it
            flag = configurable.pyre_isProtocol
        # if it doesn't know how to answer the question
        except AttributeError:
            # it isn't
            flag = False
        # all done
        return flag


# end of file
