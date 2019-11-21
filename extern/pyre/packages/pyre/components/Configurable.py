# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import itertools, textwrap


# superclass
from ..framework.Dashboard import Dashboard


# class declaration
class Configurable(Dashboard):
    """
    The base class for components and interfaces

    This class provides storage for class attributes and a place to park utilities that are
    common to both components and interfaces
    """

    # types
    from .exceptions import (
        FrameworkError, CategoryMismatchError, TraitNotFoundError, ConfigurationError,
        ProtocolCompatibilityError, ResolutionError
        )


    # framework data; every class record gets a fresh set of these values
    pyre_pedigree = None # my ancestors that are configurables, in mro
    pyre_localTraits = () # the traits explicitly specified in my declaration
    pyre_inheritedTraits = () # the traits inherited from my superclasses
    pyre_namemap = None # the map of trait aliases to their canonical names
    pyre_traitmap = None # the map of trait names to trait descriptors
    pyre_locator = None # the location of the configurable declaration
    pyre_internal = True # mark this configurable as not visible to end users

    pyre_isProtocol = False
    pyre_isComponent = False


    # basic support for the help system
    def pyre_help(self, indent=' '*4, **kwds):
        """
        Hook for the application help system
        """
        # my summary
        yield from self.pyre_showSummary(indent=indent, **kwds)
        # my public state
        yield from self.pyre_showConfigurables(indent=indent, **kwds)
        # all done
        return


    def pyre_showConfiguration(self, indent='', deep=False):
        """
        Traverse my configuration tree and display trait metadata
        """
        # prepare the indent levels
        one = indent + ' '*2
        two = one + ' '*2
        three = two + ' '*2
        # sign on
        yield "{}{.pyre_name}:".format(indent, self)

        # determine how deep to go
        traits = self.pyre_configurables() if deep else self.pyre_localConfigurables()

        # go through my traits
        for trait in traits:
            # grab the tip
            tip = trait.tip or trait.doc
            # the default value
            default = trait.default
            # and the current value
            value = getattr(self, trait.name)

            # show me
            yield "{}{.name}:".format(one, trait)
            yield "{}schema: {}".format(two, trait.typename)
            yield "{}value: {}".format(two, value)
            yield "{}default: {}".format(two, default)
            yield "{}tip: {}".format(two, tip)

            # if the value itself is a configurable
            if isinstance(value, Configurable):
                # mark it
                yield "{}configuration:".format(two)
                # and describe it
                yield from value.pyre_showConfiguration(indent=three)
        # all done
        return


    def pyre_renderConfiguration(self, deep=True):
        """
        Traverse my configuration and represent it in a JSON friendly way
        """
        # prime the document
        doc = {}
        # meta-data about all my traits
        traits = {}
        # value information about my properties
        properties = {}
        # and my components
        components = {}

        # add my state
        doc['name'] = self.pyre_name
        doc['schema'] = self.pyre_family()
        doc['value'] = self.pyre_family()
        doc['doc'] = self.__doc__
        doc['traits'] = traits
        doc['properties'] = properties
        doc['components'] = components

        # get my key
        key = self.pyre_key
        # get my scope
        scope = list(self.pyre_nameserver.getInfo(key).split)

        # determine how deep to go
        configurables = self.pyre_configurables() if deep else self.pyre_localConfigurables()

        # go through my traits
        for trait in configurables:
            # meta-data:
            # the trait name
            traitName = trait.name
            # any aliases
            traitAliases = trait.aliases
            # the schema
            traitCategory = trait.category
            traitType = trait.typename
            # and the documentation
            traitTip = trait.tip
            traitDoc = trait.doc

            # initialize the trait description
            traits[traitName] = {
                'name': traitName,
                'scope': scope,
                'aliases': list(traitAliases),
                'category': traitCategory,
                'schema': traitType,
                'doc': traitDoc,
                'tip': traitTip,
            }

            # now, for the tricky part
            value = getattr(self, traitName)

            # with facilities
            if trait.isFacility:
                # if there is something bound to the facility, ask it to describe itself
                components[traitName] = {
                    'name': value.pyre_name,
                    'value': value.pyre_name,
                }
                # and move on
                continue

            # with properties, attach the property meta-data
            properties[traitName] = {
                'name': traitName,
                'value': trait.json(value),
                'default': trait.default,
            }

        # all done
        return doc


    def pyre_showSummary(self, indent, **kwds):
        """
        Generate a short description of what I do
        """
        # if i have a docstring (and i really really should...)
        if self.__doc__:
            # format and return
            yield textwrap.indent(textwrap.dedent(self.__doc__), indent)
        # all done
        return


    def pyre_showConfigurables(self, indent='', **kwds):
        """
        Generate a description of my configurable state
        """
        # my public state
        public = []
        # collect them
        for trait in self.pyre_configurables():
            # get the name
            name = trait.name
            # get the type
            schema = trait.typename
            # and the tip
            tip = trait.tip or trait.doc
            # skip nameless undocumented ones
            if not name or not tip: continue
            # pile the rest
            public.append((name, schema, tip))

        # if we were able to find any trait info
        if public:
            # the {options} section
            yield 'options:'
            # figure out how much space we need
            width = max(len(name) for name,_,_ in public) + 2 # for the dashes
            # for each behavior
            for name, schema, tip in public:
                # show the details
                yield "{}{:>{}}: {} [{}]".format(indent, '--'+name, width, tip, schema)
            # leave some space
            yield ''
        # all done
        return


    def pyre_showBehaviors(self, spec, indent, **kwds):
        """
        Generate a description of my interface
        """
        # the pile of my behaviors
        behaviors = []
        # collect them
        for behavior in self.pyre_behaviors():
            # get the name
            name = behavior.name
            # get the tip
            tip = behavior.tip
            # if there is no tip, assume it is internal and skip it
            if not tip: continue
            # everything else gets saved
            behaviors.append((name, tip))

        # if we were able to find any usage information
        if behaviors:
            # the {usage} section
            yield 'usage:'
            # a banner with all the commands
            yield '{}{} [command]'.format(indent, spec)
            # leave some space
            yield ''
            # the beginning of the section with more details
            yield 'where [command] is one of'
            # figure out how much space we need
            width = max(len(name) for name,_ in behaviors)
            # for each behavior
            for behavior, tip in behaviors:
                # show the details
                yield '{}{:>{}}: {}'.format(indent, behavior, width, tip)
            # leave some space
            yield ''
            # all done
            return


    # introspection
    @classmethod
    def pyre_traits(cls):
        """
        Generate a sequence of all my trait descriptors, both locally declared and inherited.
        If you are looking for the traits declared in a particular class, use the attribute
        {cls.pyre_localTraits} instead.
        """
        # the full set of my descriptors is prepared by {Requirement} and separated into two
        # piles: my local traits, i.e. traits that were first declared in my class record, and
        # traits that i inherited
        return itertools.chain(cls.pyre_localTraits, cls.pyre_inheritedTraits)


    @classmethod
    def pyre_configurables(cls):
        """
        Generate a sequence of all my trait descriptors that are marked as configurable.
        """
        return filter(lambda trait: trait.isConfigurable, cls.pyre_traits())


    @classmethod
    def pyre_behaviors(cls):
        """
        Generate a sequence of all my trait descriptors that are marked as configurable.
        """
        return filter(lambda trait: trait.isBehavior, cls.pyre_traits())


    @classmethod
    def pyre_properties(cls):
        """
        Generate a sequence of all my trait descriptors that are marked as properties
        """
        return filter(lambda trait: trait.isProperty, cls.pyre_traits())


    @classmethod
    def pyre_facilities(cls):
        """
        Generate a sequence of all my trait descriptors that are marked as facilities
        """
        return filter(lambda trait: trait.isFacility, cls.pyre_traits())


    @classmethod
    def pyre_localConfigurables(cls):
        """
        Generate a sequence of my local trait descriptors that are marked as configurable.
        """
        return filter(lambda trait: trait.isConfigurable, cls.pyre_localTraits)


    @classmethod
    def pyre_localBehaviors(cls):
        """
        Generate a sequence of my local trait descriptors that are marked as behaviors
        """
        return filter(lambda trait: trait.isBehavior, cls.pyre_localTraits)


    @classmethod
    def pyre_localProperties(cls):
        """
        Generate a sequence of my local trait descriptors that are marked as properties
        """
        return filter(lambda trait: trait.isProperty, cls.pyre_localTraits)


    @classmethod
    def pyre_localFacilities(cls):
        """
        Generate a sequence of all my trait descriptors that are marked as facilities
        """
        return filter(lambda trait: trait.isFacility, cls.pyre_localTraits)


    @classmethod
    def pyre_trait(cls, alias):
        """
        Given the name {alias}, locate and return the associated descriptor
        """
        # attempt to normalize the given name
        try:
            canonical = cls.pyre_namemap[alias]
        # not one of my traits
        except KeyError:
            # complain
            raise cls.TraitNotFoundError(configurable=cls, name=alias)
        # got it; look through my trait map for the descriptor
        try:
            # if there, get the accessor
            trait = cls.pyre_traitmap[canonical]
        # if not there
        except KeyError:
            # we have a bug
            import journal
            firewall = journal.firewall("pyre.components")
            raise firewall.log("UNREACHABLE")

        # return the trait
        return trait


    # framework notifications
    @classmethod
    def pyre_classRegistered(cls):
        """
        Hook that gets invoked by the framework after the class record has been registered but
        before any configuration events
        """
        # do nothing
        return cls


    @classmethod
    def pyre_classConfigured(cls):
        """
        Hook that gets invoked by the framework after the class record has been configured,
        before any instances have been created
        """
        # do nothing
        return cls


    @classmethod
    def pyre_classInitialized(cls):
        """
        Hook that gets invoked by the framework after the class record has been initialized,
        before any instances have been created
        """
        # do nothing
        return cls


    # compatibility check
    @classmethod
    def pyre_isCompatible(cls, spec, fast=True):
        """
        Check whether {cls} is assignment compatible with {spec}

        Here, we confine ourselves to the part of the problem that involves assignment
        compatibility, i.e. whether {cls} provides at least the properties and behaviors
        specified by {spec}. More general compatibility notions are supported by the
        subclasses.

        If {fast} is True, this method will return as soon as it encounters the first
        incompatibility issue, without performing an exhaustive check of all traits. If {fast}
        is False, a thorough check of all traits will be performed resulting in a detailed
        compatibility report.
        """
        # get the report factory
        from .CompatibilityReport import CompatibilityReport
        # to build an empty one
        report = CompatibilityReport(cls, spec)

        # iterate over the traits of {spec}
        for hers in spec.pyre_traits():
            # if i have a trait by this name
            try:
                # grab it
                mine = cls.pyre_traitmap[hers.name]
            # if i don't, we have an incompatibility
            except KeyError:
                # build an error description
                error = cls.TraitNotFoundError(configurable=cls, name=hers.name)
                # add it to the report
                report.incompatibilities[hers].append(error)
                # if we are in fast mode, we have done enough
                if fast: return report
                # move on to the next trait
                continue

            # are the two traits instances of compatible classes?
            if not issubclass(type(mine), type(hers)):
                # build an error description
                error = cls.CategoryMismatchError(configurable=cls, target=spec, name=hers.name)
                # add it to the report
                report.incompatibilities[hers].append(error)
                # if we are in fast mode, we have done enough
                if fast: return report
                # otherwise move on to the next trait. N.B. the superfluous {continue} is here
                # in case more checking is added after this paragraph
                continue

        # all done
        return report


# end of file
