# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import itertools
from .. import tracking
# superclass
from .Inventory import Inventory


# declaration
class PrivateInventory(Inventory):
    """
    Strategy for managing the state of component classes and instances that were not given a
    publicly visible name, hence they are responsible for managing their own private state
    """

    # constants
    key = None # components with private inventories have no keys


    # slot access
    def setTraitValue(self, trait, factory, value, **kwds):
        """
        Set the value of the slot associated with {trait}
        """
        # grab the old slot
        old = self.traits[trait]
        # use the factory to make a new slot
        new = factory(value=value, current=old)
        # and attach the new one
        self.traits[trait] = new
        # all done
        return new, old


    def getTraitValue(self, trait):
        """
        Get the value associated with this {trait} descriptor
        """
        # get the slot
        slot = self.traits[trait]
        # return the value, along with empty meta-data
        return slot.value


    def getTraitLocator(self, trait):
        """
        Retrieve the location of the last assignment for this {trait}
        """
        # private inventories don't track trait meta-data (yet)
        return tracking.unknown()


    def getTraitPriority(self, trait):
        """
        Retrieve the priority of the last assignment for this {trait}
        """
        # private inventories don't track trait meta-data (yet)
        return None


    def getSlots(self):
        """
        Return an iterable over the trait value storage
        """
        # that's what i store in my trait map
        return self.traits.values()


    # support for building component classes and instances
    @classmethod
    def initializeClass(cls, component, **kwds):
        """
        Build inventory appropriate for a component class that is not registered with the
        nameserver
        """
        # build the inventory
        inventory = cls()
        # attach it
        component.pyre_inventory = inventory
        # collect the slots
        local = cls.localSlots(component=component)
        inherited = cls.inheritedSlots(component=component)
        slots = itertools.chain(local, inherited)
        # and populate the inventory
        inventory.populate(slots=slots)

        # invoke the configuration hook
        component.pyre_classConfigured()

        # return the inventory
        return inventory


    @classmethod
    def initializeInstance(cls, instance, **kwds):
        """
        Build inventory appropriate for a component instance that is not registered with the
        nameserver
        """
        # build the inventory
        inventory = cls()
        # attach it
        instance.pyre_inventory = inventory
        # prime the slot generator
        slots = cls.instanceSlots(instance=instance)
        # and populate the inventory
        inventory.populate(slots=slots)
        # all done
        return


    @classmethod
    def configureInstance(cls, instance):
        """
        Configure a newly minted instance
        """
        # invoke the configuration hook and pass on any errors
        yield from instance.pyre_configured()
        # all done
        return


    @classmethod
    def localSlots(cls, component):
        """
        Build slots for the locally declared traits of a {component} class
        """
        # go through the traits declared locally in {component}
        for trait in component.pyre_localTraits:
            # skip the non-configurable ones
            if not trait.isConfigurable: continue
            # yield a (trait, slot) pair
            yield trait, trait.classSlot(value=trait.default)
        # all done
        return


    @classmethod
    def inheritedSlots(cls, component):
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
                    # no worries; it must have been seen while processing a  closer ancestor
                    continue
                # otherwise, remove it from the target list
                traits.remove(trait)
                # ancestors marked {internal} do not have inventories; if this is not one of them
                if inventory is not None:
                    # get the associated slot
                    slot = inventory[trait]
                    # build a reference to it; no need to switch value processors here, since
                    # the type of an inherited trait is determined by the nearest ancestor that
                    # declared it
                    ref = slot.ref(preprocessor=trait.classSlot.pre,
                                   postprocessor=trait.classSlot.post)
                    # and yield the trait, reference slot pair
                    yield trait, ref
                # otherwise
                else:
                    # act like we own it
                    yield trait, trait.classSlot(value=trait.default)
            # if we have exhausted the trait pile
            if not traits:
                # skip the rest of the ancestors
                break
        # if we ran out of ancestors before we ran out of traits
        else:
            # complain
            missing = ', '.join(f"'{trait.name}'" for trait in traits)
            msg = f"{component}: could not locate slots for the following traits: {missing}"
            # by accessing the journal package
            import journal
            # and raising a firewall, since this is almost certainly a bug
            raise journal.firewall("pyre.components").log(msg)

        # otherwise, we are done
        return


    @classmethod
    def instanceSlots(cls, instance):
        """
        Build slots for the initial inventory of an instance by building references to all the
        slots in the inventory of its class
        """
        # get the class record of this {instance}
        component = type(instance)
        # go through all the configurable traits in {component}
        for trait in component.pyre_configurables():
            # ask the class inventory for the default value
            value = component.pyre_inventory[trait].value
            # get the instance slot factory
            factory = trait.instanceSlot
            # make a slot
            slot = factory(value=value)
            # hand the trait and the default value from the class record
            yield trait, slot
        # all done
        return


    def __str__(self):
        return f"private inventory at {id(self):#x}"


# end of file
