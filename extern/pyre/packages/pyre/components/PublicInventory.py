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
        # N.B: the previous implementation used the client key to hash the trait name directly;
        # this method suffers from a fundamental flaw: there is no easy way to build the slot
        # name. the current implementation builds the name path to the node and hands that to
        # the nameserver, which is perfectly able of building a key and assembling the name

        # get the name path of my client
        split = list(self.fragments)
        # add the trait name
        split.append(trait.name)
        # get the nameserver
        nameserver = self.pyre_nameserver
        # adjust the model
        _, new, old = nameserver.insert(split=split, **kwds)

        # all done
        return new, old


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
        # get the executive
        executive = component.pyre_executive
        # register the class with the executive
        key = executive.registerComponentClass(family=family, component=component)

        # build the inventory
        inventory = cls(key=key)
        # attach it
        component.pyre_inventory = inventory

        # collect the slots
        local = cls.localSlots(key=key, component=component)
        inherited = cls.inheritedSlots(key=key, component=component)
        slots = itertools.chain(local, inherited)
        # register them with the nameserver
        slots = cls.registerSlots(key=key, slots=slots, locator=component.pyre_locator)
        # and populate the inventory
        inventory.populate(slots=slots)

        # configure the class
        component.pyre_configurator.configureComponentClass(component=component)
        # invoke the configuration hook
        component.pyre_classConfigured()

        # return the inventory
        return inventory


    @classmethod
    def initializeInstance(cls, instance, name, implicit):
        """
        Build inventory appropriate for a component instance that has a publicly visible name and
        is registered with the nameserver
        """
        # N.B.: for EARLY BINDING
        # if this initialization is marked {implicit}, we are already in the process of
        # registering this instance; we must avoid causing the executive from re-registering it
        # because it screws up the node priorities, among other potential problems
        #
        # check for reentry
        # if implicit:
            # a key already exists; grab it
            # key = cls.pyre_nameserver.hash(name=name)
        # otherwise
        # else:
            # have the executive make a key
            # key = cls.pyre_executive.registerComponentInstance(instance=instance, name=name)
        #
        # For LATE BIDING, {implicit} is irrelevant because the instantiation happens only once

        # have the executive make a key
        key = cls.pyre_executive.registerComponentInstance(instance=instance, name=name)
        # create the inventory
        inventory = cls(key=key)
        # attach it
        instance.pyre_inventory = inventory
        # build the instance slots
        slots = cls.instanceSlots(key=key, instance=instance)
        # register them
        slots = cls.registerSlots(key=key, slots=slots, locator=instance.pyre_locator)
        # and populate
        inventory.populate(slots=slots)
        # all done
        return


    @classmethod
    def configureInstance(cls, instance):
        """
        Configure a newly minted {instance}
        """
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
        # if there are no inherited traits
        if not traits:
            # there is nothing to do
            return

        # go through each of the ancestors of {component}
        for ancestor in component.pyre_pedigree[1:]:
            # get the inventory of the ancestor
            inventory = ancestor.pyre_inventory
            # go through its configurable traits
            for trait in ancestor.pyre_configurables():
                # if the trait is not in the target pile
                if trait not in traits:
                    # no worries; it must have been seen while processing a closer ancestor
                    continue
                # otherwise, remove it from the target list
                traits.remove(trait)
                # ancestors that are marked {internal} do not get inventories; if this is not
                # one of them
                if inventory is not None:
                    # get the associated slot
                    slot = inventory[trait]
                    # build a reference to it; no need to switch value processors here, since
                    # the type of an inherited trait is determined by the nearest ancestor that
                    # declared it
                    ref = slot.ref(key=key[trait.name],
                                   preprocessor=trait.classSlot.pre,
                                   postprocessor=trait.classSlot.post)
                    # yield the trait, its class slot factory, and a reference to the inherited
                    # slot
                    yield trait, trait.classSlot, ref
                # otherwise
                else:
                    # act like we own it
                    yield trait, trait.classSlot, trait.default

            # if we have exhausted the trait pile
            if not traits:
                # skip the rest of the ancestors
                break
        # if we ran out of ancestors before we ran out of traits
        else:
            # complain
            missing = ', '.join(f"'{trait.name}'" for trait in traits)
            msg = f"{component}: could not locate slots for the following traits: {missing}"
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
            # get its value
            value = slot.value
            # grab the trait slot factory
            factory = trait.instanceSlot
            # hand the trait, the factory and the default value from the class record
            yield trait, factory, value
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
        # look up the base name
        base = nameserver.getName(key)
        # go through the (trait, slot) pairs
        for trait, factory, value in slots:
            # get the name of the trait descriptor
            name = trait.name
            # build the trait full name
            fullname = nameserver.join(base, name)
            # place the slot with the nameserver
            traitKey, _, _ = nameserver.insert(name=fullname, value=value,
                                               factory=factory,
                                               locator=locator, priority=priority())
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
        return f"public inventory at {id(self):#x}"


# end of file
