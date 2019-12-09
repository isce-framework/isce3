# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import collections, itertools
from .. import tracking
# superclass
from .Requirement import Requirement


# class declaration
class Actor(Requirement):
    """
    The metaclass of components

    {Actor} takes care of all the tasks necessary to convert a component declaration into a
    configurable entity that coöperates fully with the framework
    """


    # types
    from .Role import Role
    from .PublicInventory import PublicInventory
    from .PrivateInventory import PrivateInventory
    from .exceptions import ImplementationSpecificationError, ProtocolError


    # public data
    @property
    def pyre_name(self):
        """
        Return the component's family name
        """
        return self.pyre_family()


    # meta-methods
    def __new__(cls, name, bases, attributes, *, family=None, implements=None, **kwds):
        """
        Build a new component record

        parameters:
            {cls}: the metaclass invoked; guaranteed to be a descendant of {Actor}
            {name}, {bases}, {attributes}: the usual class specification
            {implements}: the tuple of protocols that this component is known to implement
        """
        # build the protocol specification
        protocol = cls.pyre_buildProtocol(name, bases, implements)

        # build and add the protocol specification to the attributes
        attributes["pyre_implements"] = protocol
        # save the location of the component declaration
        # N.B.: the locator might be incorrect if the metaclass hierarchy gets deeper;
        # e.g. {pyre.shells.Director}, which doesn't override {__new__} so it is not affected.
        attributes["pyre_locator"] = tracking.here(1)

        # get my ancestors to build the class record
        component = super().__new__(cls, name, bases, attributes, **kwds)

        # and pass the component on
        return component


    def __init__(self, name, bases, attributes, *, family=None, **kwds):
        """
        Initialize a new component class record
        """
        # initialize the record
        super().__init__(name, bases, attributes, **kwds)

        # if this is an internal component, there is nothing further to do
        if self.pyre_internal: return

        # pick the appropriate inventory strategy
        inventory = self.PublicInventory if family else self.PrivateInventory
        # and invoke it to register the component class and build its inventory
        inventory.initializeClass(component=self, family=family)

        # get my protocol specification
        protocol = self.pyre_implements
        # if one was derivable from the declaration, check protocol compatibility
        if protocol:
            # generate a compatibility report
            report = self.pyre_isCompatible(protocol)
            # if it's not a clean sheet
            if not report:
                # complain
                raise self.ProtocolError(self, protocol, report)

        # register with the component registrar
        self.pyre_registrar.registerComponentClass(component=self)
        # invoke the registration hook
        self.pyre_classRegistered()

        # all done
        return


    def __call__(self, name=None, locator=None, globalAliases=False, **kwds):
        """
        Build an instance of one of my classes
        """
        # ask the component class for any opinions on the name of this instance
        name = self.pyre_normalizeInstanceName(name)
        # get the registrar
        registrar = self.pyre_registrar

        # if I know the name
        if name:
            # look for this name among my instances
            instance = registrar.retrieveComponentByName(componentClass=self, name=name)
            # if found, return it
            if instance: return instance
            # otherwise, we are making a new instance under this name; if i were asked to alias
            # its traits globally
            if globalAliases:
                # get the nameserver
                nameserver = self.pyre_nameserver
                # my configurable traits
                traits = self.pyre_configurables()
                # build the set of names
                aliases = { alias for trait in traits for alias in trait.aliases }
                # merge global settings
                nameserver.pullGlobalIntoScope(scope=name, symbols=aliases)
        # if not
        else:
            # ask the component registrar for help
            name = registrar.nameInstance(componentClass=self)

        # in any case, record the caller's location
        locator = tracking.here(1) if locator is None else locator
        # build the instance
        instance = super().__call__(name=name, locator=locator, **kwds)

        # invoke the instantiation hook and harvest any errors
        initializationErrors = list(instance.pyre_initialized())
        # attach them
        instance.pyre_initializationErrors = initializationErrors

        # if there were any
        if initializationErrors:
            # complain
            raise instance.ConfigurationError(configurable=instance, errors=initializationErrors)

        # register with the component registrar
        registrar.registerComponentInstance(instance=instance)
        # invoke the registration hook
        instance.pyre_registered()

        # and return it
        return instance


    def __setattr__(self, name, value):
        """
        Trap attribute setting in my class record instances to support setting the default
        value using the natural syntax
        """
        # print(f"Actor.__setattr__: '{name}'<-'{value}'")
        # in the early states of decorating component class records, it is important to
        # recognize internal attributes
        if name.startswith('pyre_'):
            # and process them normally without attempting to lookup the attribute name in the
            # name map, which might not have been built yet
            return super().__setattr__(name, value)
        # for the rest, try to
        try:
            # normalize the name
            name = self.pyre_namemap[name]
        # if it isn't one of my traits
        except KeyError:
            # treat as a normal assignment
            return super().__setattr__(name, value)

        # the name refers to one of my traits; find it
        trait = self.pyre_traitmap[name]
        # build an appropriate locator
        locator = tracking.here(level=1)
        # set the priority
        priority = self.pyre_executive.priority.explicit()
        # set the value
        self.pyre_inventory.setTraitValue(
            trait=trait, factory=trait.classSlot,
            value=value, priority=priority, locator=locator)
        # and return
        return


    def __str__(self):
        # get my family name
        family = self.pyre_family()
        # if i have one
        if family:
            # use it
            return f"component '{family}'"
        # otherwise, use my class name
        return f"component '{self.__name__}'"


    # implementation details
    @classmethod
    def pyre_buildProtocol(cls, name, bases, implements):
        """
        Build a class that describes the implementation requirements imposed on this
        {component}, given its class record and the list of protocols it {implements}
        """
        # try to understand what the component author specified
        if implements is not None:
            # if {implements} is a single protocol
            if cls.pyre_isProtocol(implements):
                # make a pile with one entry
                mine = implements,
            # the only legal alternative is an iterable of {Protocol} subclasses
            elif isinstance(implements, collections.abc.Iterable):
                # go through its contents and collect the ones that are not protocols
                errors = tuple(itertools.filterfalse(cls.pyre_isProtocol, implements))
                # if there were any
                if errors:
                    # complain
                    raise cls.ImplementationSpecificationError(name=name, errors=errors)
                # otherwise, they were all protocols; put them on a pile
                mine = tuple(implements)
            # in any other case
            else:
                # the entire specification is unrecognizable, so complain
                raise cls.ImplementationSpecificationError(name=name, errors=[implements])
        # otherwise
        else:
            # i don't have an implementation specification
            mine = ()

        # now, add the commitments made by my immediate ancestors
        inherited = tuple(
            base.pyre_implements
            for base in bases
            if cls.pyre_isComponent(base) and base.pyre_implements is not None)

        # assemble the full set
        protocols = mine + inherited
        # bail out if we didn't manage to find any protocols
        if not protocols: return None
        # if there is only one protocol on my pile
        if len(protocols) == 1:
            # use it directly
            return protocols[0]
        # otherwise, derive a protocol from the harvested ones and return it as the
        # implementation specification
        return cls.Role("protocol", protocols, dict(), internal=True)


# end of file
