# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import operator
import itertools
from .. import tracking
# superclass
from .Inventory import Inventory


# declaration
class PublicInventory(Inventory):
    """
    Strategy for providing access to the state of component classes and instances that were
    given a publicly visible name and use slots managed the pyre nameserver for storage.
    """


    # public data
    @property
    def name(self):
        """
        Look up the family name of my client
        """
        # get the nameserver
        nameserver = self.pyre_nameserver
        # ask it for my full name and return it
        return nameserver.getName(key=self.key)


    @property
    def fragments(self):
        """
        Look up the family name of my client
        """
        # get the nameserver
        nameserver = self.pyre_nameserver
        # ask it for my full name and return it
        return nameserver.getSplitName(key=self.key)


    @property
    def package(self):
        """
        Return the package associated with this client
        """
        # get the nameserver
        nameserver = self.pyre_nameserver
        # ask it for my full name
        name = nameserver.getName(key=self.key)
        # split it apart; the package name is the zeroth entry
        packageName = nameserver.split(name)[0]
        # use the name to look up the package
        return nameserver[packageName]


    # slot access
    def setTraitValue(self, trait, **kwds):
        """
        Set the value of the slot associated with the given {trait} descriptor
        """
        # hash the trait name
        key = self.key[trait.name]
        # get the nameserver
        nameserver = self.pyre_nameserver
        # adjust the model
        nameserver.insert(key=key, **kwds)
        # all done
        return


    def getTraitValue(self, trait):
        """
        Get the value associated with this {trait} descriptor
        """
        # hash the trait name
        key = self.key[trait.name]
        # access the nameserver
        nameserver = self.pyre_nameserver
        # ask it for the slot
        slot = nameserver.getNode(key)
        # return the value
        return slot.value


    def getTraitLocator(self, trait):
        """
        Get the location of last assignment for this {trait}
        """
        # hash the trait name
        key = self.key[trait.name]
        # access the nameserver
        nameserver = self.pyre_nameserver
        # ask it for the meta-data
        info = nameserver.getInfo(key)
        # return the locator
        return info.locator


    def getTraitPriority(self, trait):
        """
        Retrieve the priority of the last assignment for this {trait}
        """
        # hash the trait name
        key = self.key[trait.name]
        # access the nameserver
        nameserver = self.pyre_nameserver
        # ask it for the meta-data
        info = nameserver.getInfo(key)
        # return the priority
        return info.priority


    def getSlots(self):
        """
        Return an iterable over the trait value storage
        """
        # access the nameserver
        nameserver = self.pyre_nameserver
        # go through the slot keys
        for key in self.traits.values():
            # look up the slot and send it off
            yield nameserver.getNode(key)
        # all done
        return



    # support for constructing component classes and instances
    @classmethod
    def initializeClass(cls, component, family, **kwds):
        """
        Build inventory appropriate for a component instance that has a publicly visible name and
        is registered with the nameserver
        """
        # register the class with the executive
        key = component.pyre_executive.registerComponentClass(family=family, component=component)

        # collect the slots
        local = cls.localSlots(key=key, component=component)
        inherited = cls.inheritedSlots(key=key, component=component)
        slots = itertools.chain(local, inherited)

        # register them with the nameserver
        slots = cls.registerSlots(key=key, slots=slots, locator=component.pyre_locator)

        # build the inventory
        inventory = cls(key=key, slots=slots)
        # attach it
        component.pyre_inventory = inventory

        # configure the class
        component.pyre_configurator.configureComponentClass(component=component)
        # invoke the configuration hook
        component.pyre_classConfigured()

        # return the inventory
        return inventory


    @classmethod
    def initializeInstance(cls, instance, name):
        """
        Build inventory appropriate for a component instance that has a publicly visible name and
        is registered with the nameserver
        """
        # have the executive make a key
        key = cls.pyre_executive.registerComponentInstance(instance=instance, name=name)

        # build the instance slots
        slots = cls.instanceSlots(key=key, instance=instance)
        # register them
        slots = cls.registerSlots(key=key, slots=slots, locator=instance.pyre_locator)
        # build the inventory out of the instance slots and attach it
        instance.pyre_inventory = cls(key=key, slots=slots)
        # configure the instance
        cls.pyre_configurator.configureComponentInstance(instance=instance)
        # invoke the configuration hook and pass on any errors
        yield from instance.pyre_configured()
        # all done
        return


    @classmethod
    def localSlots(cls, key, component):
        """
        Build slots for the locally declared traits of a {component} class
        """
        # go through the traits declared locally in {component}
        for trait in component.pyre_localTraits:
            # skip the non-configurable ones
            if not trait.isConfigurable: continue
            # yield the trait, its class slot factory, and the default value of the trait
            yield trait, trait.classSlot, trait.default
        # all done
        return


    @classmethod
    def inheritedSlots(cls, key, component):
        """
        Build slots for the inherited traits of a {component} class
        """
        # collect the traits I am looking for
        traits = set(trait for trait in component.pyre_inheritedTraits if trait.isConfigurable)
        # if there are no inherited traits, bail out
        if not traits: return
        # go through each of the ancestors of {component}
        for ancestor in component.pyre_pedigree[1:]:
            # and all its configurable traits
            for trait in ancestor.pyre_configurables():
                # if the trait is not in the target pile
                if trait not in traits:
                    # no worries; it must have been seen while processing a closer ancestor
                    continue
                # otherwise, remove it from the target list
                traits.remove(trait)
                # get the associated slot
                slot = ancestor.pyre_inventory[trait]
                # build a reference to it; no need to switch postprocessor here, since the type of
                # an inherited trait is determined by the nearest ancestor that declared it
                ref = slot.ref(key=key[trait.name], postprocessor=trait.classSlot.processor)
                # yield the trait, its class slot factory, and a reference to the inherited trait
                yield trait, trait.classSlot, ref
            # if we have exhausted the trait pile
            if not traits:
                # skip the rest of the ancestors
                break
        # if we ran out of ancestors before we ran out of traits
        else:
            # complain
            missing = ', '.join('{!r}'.format(trait.name) for trait in traits)
            msg = "{}: could not locate slots for the following traits: {}".format(
                component, missing)
            # by raising a firewall, since this is almost certainly a bug
            import journal
            raise journal.firewall("pyre.components").log(msg)

        # otherwise, we are done
        return


    @classmethod
    def instanceSlots(cls, key, instance):
        """
        Build slots for the initial inventory of an instance by building references to all the
        slots in the inventory of its class
        """
        # get the component class of this {instance}
        component = type(instance)
        # go through all the configurable traits in {component}
        for trait in component.pyre_configurables():
            # ask the class inventory for the slot that corresponds to this trait
            slot = component.pyre_inventory[trait]
            # build a reference to the class slot
            ref = slot.ref(key=key[trait.name], postprocessor=trait.instanceSlot.processor)
            # hand the trait, slot and its value
            yield trait, trait.instanceSlot, ref
        # all done
        return


    @classmethod
    def registerSlots(cls, key, slots, locator):
        """
        Go through the (trait, factory, value) tuples in {slots} and register them with the
        nameserver
        """
        # get the nameserver
        nameserver = cls.pyre_nameserver
        # get the factory of priorities in the {defaults} category
        priority = nameserver.priority.defaults
        # look up the basename
        base = nameserver.getName(key)
        # go through the (trait, slot) pairs
        for trait, factory, value in slots:
            # get the name of the trait descriptor
            name = trait.name
            # build the trait full name
            fullname = nameserver.join(base, name)
            # place the slot with the nameserver
            traitKey = nameserver.insert(name=fullname, value=value,
                                         factory=factory, locator=locator, priority=priority())
            # register the trait aliases
            for alias in trait.aliases:
                # skip the canonical name
                if alias == name: continue
                # notify the nameserver
                nameserver.alias(base=key, alias=alias, target=traitKey)
            # hand this (trait, key) pair to the caller
            yield trait, traitKey
        # all done
        return


    # meta-methods
    def __init__(self, key, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the key
        self.key = key
        # all done
        return


    def __getitem__(self, trait):
        """
        Retrieve the slot associated with {trait}
        """
        # get the key
        key = super().__getitem__(trait)
        # ask the nameserver for the slot and return it
        return self.pyre_nameserver.getNode(key)


    def __str__(self):
        return "public inventory at {:#x}".format(id(self))


# end of file
