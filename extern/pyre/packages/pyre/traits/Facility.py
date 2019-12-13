# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclasses
from .. import schemata
from .Slotted import Slotted


# declaration
class Facility(Slotted, schemata.component):
    """
    The descriptor for traits that are components
    """

    # Facility is faced with the following problem: the expected results of coercing are
    # different depending on whether the object whose trait is being processed is a component
    # class or a component instance. In the latter case, we want to cast the trait value into
    # an actual component instance that is compatible with the facility requirements; in the
    # former we are happy with either a compatible component declaration or an instance.

    # framework data
    category = 'component'
    # predicate that indicates whether this trait is a facility
    isFacility = True


    # public data
    @property
    def default(self):
        """
        Access to my default value
        """
        # get my default value
        default = self._default
        # if it is still at its trivial value
        if default is schemata.component.default:
            # ask my protocol
            return self.protocol.pyre_default
        # otherwise, return it
        return default


    @default.setter
    def default(self, value):
        """
        Set my default value
        """
        # save {value} as the default
        self._default = value
        # all done
        return


    def macro(self, **kwds):
        """
        Return the default strategy for handling expressions in slot values
        """
        # build expressions
        return self.pyre_nameserver.expression(**kwds)


    def native(self, **kwds):
        """
        The strategy for building slots from more complex input values
        """
        # facility values are held in variables
        return self.pyre_nameserver.variable(**kwds)


    # interface
    def instantiate(self, value, node, incognito=False, **kwds):
        """
        Coerce {value} into an instance of a component compatible with my protocol
        """
        # run the value through my regular coercion
        value = self.process(value=value, node=node, **kwds)
        # if {value} results in {None} after initial processing, leave it alone too
        if value is None: return None
        # if what I got back is a component instance, we are all done
        if isinstance(value, self.protocol.component): return value

        # get the key of the node
        key = node.key
        # if it has one
        if key:
            # get the nameserver
            nameserver = self.pyre_nameserver
            # decide what I am supposed to name the new component
            name = nameserver.getName(key) if not incognito else None
            # and get the locator from the key metadata
            locator = nameserver.getInfo(key).locator
        # otherwise
        else:
            # we have no name
            name = None
            # and no locator
            locator = None

        # instantiate and return; mark as implicit to denote that this instance was created
        # during facility processing, rather than explicitly
        return value(name=name, locator=locator, implicit=True)


    def render(self, value, renderer, workload, incognito=False):
        """
        Render {value} using renderer
        """
        # get the name to use
        name = None if incognito else self.name
        # delegate to my protocol
        yield from self.protocol.pyre_render(renderer=renderer,
                                             name=name, component=value, workload=workload)
        # all done
        return


    # meta-methods
    def __init__(self, protocol, **kwds):
        # chain up
        super().__init__(protocol=protocol, **kwds)
        # build my slot factories
        self.classSlot = self.factory(trait=self, post=self.process)
        # self.instanceSlot = self.factory(trait=self, pre=self.instantiate, post=self.instantiate)
        self.instanceSlot = self.factory(trait=self, post=self.instantiate)
        # add the converter from my protocol to my pile
        self.converters.append(protocol.pyre_convert)
        # repeat with the normalizer
        self.normalizers.append(protocol.pyre_normalize)
        # and the validator
        self.validators.append(protocol.pyre_validate)
        # all done
        return


    def __str__(self):
        return "{0.name!r}: a facility with {0.protocol}".format(self)


# end of file
