# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access to the default trait
from .Property import Property
# superclass
from .Slotted import Slotted


# declaration
class Dict(Slotted):
    """
    A property that maps strings to components
    """

    # constants
    typename = 'dict' # the name of my type


    # public data
    @property
    def macro(self):
        """
        The default strategy for handling slot values that are strings and therefore subject to
        some kind of evaluation in the context of the configuration store
        """
        # whatever my schema says
        return self.schema.macro


    def native(self, value, **kwds):
        """
        The default strategy for handling macros in slot values
        """
        # if the value is any kind of mapping object
        if isinstance(value, collections.abc.Mapping):
            # build a dictionary of nodes
            nodes = {key: self.schema.macro(value=load) for key, load in value.items()}
            # build a mapping node and return it
            return self.pyre_nameserver.mapping(nodes, **kwds)

        # if the value is any kind of iterable
        if isinstance(value, collections.abc.Iterable):
            # build a dictionary of nodes
            nodes = {key: self.schema.macro(value=load) for key, load in value}
            # build a mapping node and return it
            return self.pyre_nameserver.mapping(nodes, **kwds)

        # if it is {None}
        if value is None:
            # store it in a variable
            return self.pyre_nameserver.variable(value=value, **kwds)

        # shouldn't get here
        assert False, 'unreachable'


    # my value processors
    def process(self, value, **kwds):
        """
        Walk {value} through the casting procedure appropriate for clients that are component
        classes
        """
        # leave {None} alone
        if value is None: return None
        # make sure we are building class slots, and delegate
        return self.catalog(factory=self.schema.classSlot, value=value, **kwds)


    def instantiate(self, value, **kwds):
        """
        Walk {value} through the casting procedure appropriate for clients that are component
        instances
        """
        # leave {None} alone
        if value is None: return None
        # make sure we are building instance slots, and delegate
        return self.catalog(factory=self.schema.instanceSlot, value=value, **kwds)


    # framework hooks triggered by client configuration
    def classConfigured(self, component, **kwds):
        """
        Notification that the client class record has been configured
        """
        # chain up
        super().classConfigured(component=component, **kwds)
        # configure the class record
        self.configureClient(client=component,
                             myFactory=self.classSlot, traitFactory=self.schema.classSlot)
        # all done
        return self


    def instanceConfigured(self, instance, **kwds):
        """
        Notification that the client class record has been configured
        """
        # chain up
        super().instanceConfigured(instance=instance, **kwds)
        # configure the instance
        self.configureClient(client=instance,
                             myFactory=self.instanceSlot, traitFactory=self.schema.instanceSlot)
        # all done
        return self


    # meta-methods
    def __init__(self, schema=Property.identity(), default=object, **kwds):
        # adjust the default value
        default = dict() if default is object else default
        # chain up with a potentially adjusted default value
        super().__init__(default=default, **kwds)
        # record my schema
        self.schema = schema
        # build my slot factories
        self.classSlot = self.factory(trait=self, processor=self.process)
        self.instanceSlot = self.factory(trait=self, processor=self.instantiate)
        # all done
        return


    # implementation details
    # catalog initialization
    def catalog(self, factory, value, node, **kwds):
        """
        Instantiate and initialize an appropriate map
        """
        # get the node key
        key = node.key
        # grab my schema
        schema = self.schema
        # decide which mapping strategy to use: if the {node} has no key
        if key is None:
            #  make a {NameMap}
            catalog = NameMap(schema=schema, factory=factory)
        # otherwise
        else:
            # make a key based map
            catalog = KeyMap(schema=schema, factory=factory, key=key)

        # use this priority
        priority = self.pyre_nameserver.priority.user
        # go through the entries in {value}
        for key, setting in value.items():
            # make a locator
            locator = tracking.simple('while adding entry {!r} to {.name!r}'.format(key, self))
            # and update my map
            catalog.insert(name=key, value=setting, locator=locator, priority=priority())
        # and return it
        return catalog


    # client configuration
    def configureClient(self, client, myFactory, traitFactory):
        """
        A named client with public inventory requires further configuration
        """
        # access the nameserver
        nameserver = self.pyre_nameserver
        # and the configurator
        configurator = self.pyre_configurator
        # get my schema
        schema = self.schema
        # find my slot
        slot = client.pyre_inventory[self]
        # this gets called only for public inventory items, so I am guaranteed a key
        key = slot.key
        # get my name
        tag = nameserver.getName(key)
        # the priority of all these assignments
        userPriority = nameserver.priority.user

        # make a key based map
        catalog = KeyMap(schema=schema, factory=traitFactory, key=key)

        # grab all direct assignments to this key
        for name, node in configurator.retrieveDirectAssignments(key):
            # extract the item key
            name = nameserver.split(name)[-1]
            # and its value
            value = node.value
            # make a locator
            locator = tracking.simple('while adding entry {!r} to {.name!r}'.format(name, self))
            # and store them
            catalog.insert(name=name, value=value, priority=userPriority(), locator=locator)

        # grab all deferred assignments to this key
        for assignment, priority in configurator.retrieveDeferredAssignments(key):
            # store them
            catalog.insert(
                name=assignment.key[0], value=assignment.value,
                priority=priority, locator=assignment.locator)

        # get the my current slot value
        current = slot.value
        # if non-trivial, use it to initialize my catalog; i expect it to be a dictionary
        # this must happen after direct and indirect assignments to avoid changing the
        # nameserver model while the update is taking place
        if current:
            # raise NotImplementedError("NYI: priorities/locators?")
            catalog.update(current)
        # one more special case: no settings in the store and {None} value
        if current is None and not catalog:
            # leave uninitialized
            catalog = None

        # make a locator
        here = tracking.simple('while configuring {.pyre_name!r}'.format(client))

        # attach my new value
        client.pyre_inventory.setTraitValue(
            trait=self, factory=myFactory,
            value=catalog, priority=userPriority(), locator=here)

        # all done
        return self


# implementation details
# externals
# superclasses
import collections.abc
from ..framework.Dashboard import Dashboard
from .. import tracking


# the helper container classes
class Map(collections.abc.MutableMapping, Dashboard):
    """
    The base class for the storage helpers
    """

    # public data
    schema = None
    factory = None # information necessary to make slots

    # meta-methods
    def __init__(self, schema, factory, *args, **kwds):
        # chain  up
        super().__init__(**kwds)
        # my storage
        self.map = {}
        # set my schema
        self.schema = schema
        # and my slot factory
        self.factory = factory
        # initialize my contents
        self.update(*args, **kwds)
        # all done
        return

    def __delitem__(self, name):
        """
        Remove {name} from my map
        """
        # easy enough
        del self.map[name]
        # all done
        return

    def __iter__(self):
        """
        Create an iterator over my map
        """
        # easy enough
        return iter(self.map)

    def __len__(self):
        """
        Compute my size
        """
        # easy enough
        return len(self.map)

    def __contains__(self, name):
        """
        Check whether {name} is in my map
        """
        # easy enough
        return name in self.map

    def __setitem__(self, name, value):
        """
        Store {value} in the map under {name}
        """
        # build a priority
        priority = self.pyre_nameserver.priority.explicit()
        # and a locator
        locator = tracking.here(-1)
        # delegate
        return self.insert(name=name, value=value, priority=priority, locator=locator)

    def __str__(self):
        """
        Build a simple string representation of my contents
        """
        return "{{{}}}".format(
            ", ".join(["{}: {}".format(key, value) for key,value in self.items()]))

    # private data
    map = None


class KeyMap(Map):
    """
    A storage strategy that is appropriate when a client has public inventory
    """


    # meta-methods
    def __init__(self, key, *args, **kwds):
        # chain  up
        super().__init__(*args, **kwds)
        # save my client's name
        self.name = self.pyre_nameserver.getName(key)
        # all done
        return


    # slot access
    def __getitem__(self, name):
        """
        Retrieve the value associated with {name} and convert according to my schema
        """
        # get the key
        key = self.map[name]
        # get the nameserver
        nameserver = self.pyre_nameserver
        # and return its value
        return nameserver[key]


    # implementation details
    def insert(self, name, value, priority, locator):
        """
        Store {value} in the map under {name}
        """
        # get the nameserver
        nameserver = self.pyre_nameserver
        # build the full name of the map entry
        fullname = nameserver.join(self.name, name)
        # insert into the model
        key = nameserver.insert(name=fullname, value=value,
                                factory=self.factory, locator=locator, priority=priority)
        # adjust my map
        self.map[name] = key
        # all done
        return


class NameMap(Map):
    """
    A storage strategy for nameless clients
    """

    def __getitem__(self, name):
        """
        Retrieve the value associated with {name} and convert according to my schema
        """
        # get the slot
        node = self.map[name]
        # and return its value
        return node.value


    # implementation details
    def insert(self, name, value, **kwds):
        """
        Build a slot to hold {value} and place it in the map
        """
        # make a slot
        new = self.factory(value=value)
        # look for an existing slot under the same name
        try:
            # get it
            old = self.map[name]
        # if there wasn't one
        except KeyError:
            # no worries
            pass
        # if there was
        else:
            # replace the old with the new in the evaluation graph
            new.replace(old)
        # update the map
        self.map[name] = new
        # all done
        return


# end of file
