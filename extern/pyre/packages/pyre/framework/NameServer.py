# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import collections
from .. import tracking # for locators
from ..traits.Property import Property as properties # to get the default trait type
# superclass
from ..calc.Hierarchical import Hierarchical


# my declaration
class NameServer(Hierarchical):
    """
    The manager of the full set of runtime objects that are accessible by name. This includes
    everything from configuration settings to components and interfaces
    """


    # types
    from .Package import Package
    # node storage and metadata
    from .Slot import Slot as node
    from .SlotInfo import SlotInfo as info
    from .Priority import Priority as priority


    # public data
    @property
    def configpath(self):
        """
        Return an iterable over my configuration path
        """
        # the answer is in my store
        return self['pyre.configpath']


    # framework object management
    def configurable(self, name, configurable, locator, priority=None):
        """
        Add {configurable} to the model under {name}
        """
        # get the right priority
        priority = self.priority.package() if priority is None else priority
        # fill out the node info
        name, split, key = self.info.fillNodeId(model=self, name=name)
        # build the node metadata
        info = self.info(name=name, split=split, key=key,
                         locator=locator, priority=priority)
        # grab the key
        key = info.key
        # build a slot to hold the {configurable}
        slot = self.variable(key=key, value=configurable)
        # store it in the model
        self._nodes[key] = slot
        self._metadata[key] = info

        # and return the key
        return key


    def package(self, name, executive, locator):
        """
        Retrieve the named package from the model. If there is no such package, instantiate one,
        configure it, and add it to the model.
        """
        # take {name} apart and extract the package name as the top level identifier
        name = self.split(name)[0]
        # hash it
        key = self._hash[name]
        # if there is a node registered under this key
        try:
            # grab it
            node = self._nodes[key]
        # if not
        except KeyError:
            # if I do not have a locator, point to my caller
            locator = tracking.here(level=1) if locator is None else locator
            # make one
            package = self.createPackage(name=name, locator=locator)
            # configure it
            package.configure(executive=executive)
        # if it's there
        else:
            # get the value of the node
            package = node.value
            # make sure it is a package, and if not
            if not isinstance(package, self.Package):
                # get the journal
                import journal
                # build the report
                complaint = 'name conflict while configuring package {!r}: {}'.format(name, package)
                # and complain
                raise journal.error('pyre.configuration').log(complaint)

        # return the package
        return package


    # auxiliary object management
    def createPackage(self, name, locator):
        """
        Build a new package node and attach it to the model
        """
        # make one
        package = self.Package(name=name, locator=locator)
        # hash the name
        key = self._hash[name]
        # get the right priority
        priority = self.priority.package()
        # attach it to a slot
        slot = self.literal(key=key, value=package)
        # fill out the node info
        name, split, key = self.info.fillNodeId(model=self, key=key, name=name)
        # store it in the model
        self._nodes[key] = slot
        self._metadata[key] = self.info(name=name, key=key, split=split,
                                        locator=package.locator, priority=priority)
        # and return it
        return package


    # expansion services
    def evaluate(self, expression):
        """
        Evaluate the given {expression} in my current context
        """
        # attempt to
        try:
            # evaluate the expression
            return self.node.expression.expand(model=self, expression=expression)
        # with empty expressions
        except self.EmptyExpressionError as error:
            # return the expanded text, since it the input may have contained escaped braces
            return error.expression


    def interpolate(self, expression):
        """
        Interpolate the given {expression} in my current context
        """
        # attempt to
        try:
            # evaluate the expression
            return self.node.interpolation.expand(model=self, expression=expression)
        # with empty expressions
        except self.EmptyExpressionError as error:
            # return the expanded text, since it the input may have contained escaped braces
            return error.expression


    # override superclass methods
    def insert(self, value, priority, locator, key=None, name=None, split=None, factory=None):
        """
        Add {value} to the store
        """
        # figure out the node info
        name, split, key = self.info.fillNodeId(model=self, key=key, split=split, name=name)

        # look for metadata
        try:
            # registered under this key
            meta = self._metadata[key]
        # if there's no registered metadata, this is the first time this name was encountered
        except KeyError:
            # if we need to build type information
            if not factory:
                # use instance slots for an identity trait, by default
                factory = properties.identity(name=name).instanceSlot
            # build the info node
            meta = self.info(name=name, split=split, key=key,
                             priority=priority, locator=locator, factory=factory)
            # and attach it
            self._metadata[key] = meta
        # if there is an existing metadata node
        else:
            # check whether this assignment is of lesser priority, in which case we just leave
            # the value as is
            if priority < meta.priority:
                # but we may have to adjust the trait
                if factory:
                    # which involves two steps: first, update the info node
                    meta.factory = factory
                    # and now look for the existing model node
                    old = self._nodes[key]
                    # so we can update its value postprocessor
                    old.postprocessor = factory.processor
                # in any case, we are done here
                return key
            # ok: higher priority assignment; check whether we should update the descriptor
            if factory: meta.factory = factory
            # adjust the locator and priority of the info node
            meta.locator = locator
            meta.priority = priority

        # if we get this far, we have a valid key, and valid and updated metadata; start
        # processing the value by getting the trait; use the info node, which is the
        # authoritative source of this information
        factory = meta.factory
        # and ask it to build a node for the value
        new = factory(key=key, value=value)

        # if we are replacing an existing node
        try:
            # get it
            old = self._nodes[key]
        # if not
        except KeyError:
            # no worries
            pass
        # otherwise
        else:
            # adjust the dependency graph
            new.replace(old)

        # place the new node in the model
        self._nodes[key] = new

        # and return
        return key


    def retrieve(self, name):
        """
        Retrieve the node registered under {name}. If no such node exists, an error marker will
        be built, stored in the symbol table under {name}, and returned.
        """
        # hash the {name}
        key = self.hash(name)
        # if a node is already registered under this key
        try:
            # grab it
            node = self._nodes[key]
        # otherwise
        except KeyError:
            # fill out the node info
            name, split, key = self.info.fillNodeId(model=self, key=key, name=name)
            # build an error marker
            node = self.node.unresolved(key=key, request=name)
            # add it to the pile
            self._nodes[key] = node
            self._metadata[key] = self.info(name=name, split=split, key=key)
        # return the node
        return node


    # implementation details
    # adding entries to the model: the highest level interface
    def __setitem__(self, name, value):
        """
        Convert {value} into a node and update the model
        """
        # figure out the location of my caller
        locator = tracking.here(1)
        # make a priority ranking from the explicit category
        priority = self.priority.explicit()

        # add the value to the model
        return self.insert(name=name, value=value, priority=priority, locator=locator)


    # handling of content and topological changes to the store
    def store(self, key, name, node, info):
        """
        Associate {name}, {node} and {info} with {key}
        """
        # adjust the node key
        node.key = key
        # attach the node
        self._nodes[key] = node
        # adjust the metadata
        info.key = key
        info.name = name
        # and register it under key
        self._metadata[key] = info
        # all done
        return


    def replace(self, key, name, oldNode, oldInfo, newNode, newInfo):
        """
        Choose which settings to retain
        """
        # if the new node has higher priority
        if newInfo.priority > oldInfo.priority:
            # replace the old node in its evaluation graph
            newNode.replace(oldNode)
            # adjust the key in the new node
            newNode.key = key
            # adjust the post-processor
            newNode.postprocessor = oldNode.postprocessor
            # attach it to the store
            self._nodes[key] = newNode
            # adjust the metadata
            oldInfo.priority = newInfo.priority
            oldInfo.locator = newInfo.locator

        # all done
        return


    # aliasing
    def pullGlobalIntoScope(self, scope, symbols):
        """
        Merge settings for {traits} between global scope and the scope of {name}
        """
        # get the global scope
        top = self._hash
        # build the scope of the new instance
        key = self.hash(scope)
        # with each one
        for symbol in symbols:
            # if the name has never been hashed
            if symbol not in top:
                # nothing to do; no assignments have been made to it or its possible children
                continue
            # if the scope and symbol are identical
            if scope == symbol:
                # merging will fail, so skip it before we damage the symbol table
                continue
            # build the destination key
            destination = key[symbol]
            # make an alias
            source = top.alias(alias=symbol, target=destination)
            # construct the global name for the symbol
            canonical = self.join(scope, symbol)
            # and try to
            try:
                # to merge the information
                self.merge(source=source, canonical=canonical, destination=destination, name=symbol)
            # if this fails
            except Exception:
                # there must be some residual naming conflict between this trait and a global
                # object; issue a warning and ignore
                continue

        # all done
        return


    # meta-methods
    def __init__(self, name='pyre::nameserver', **kwds):
        # chain up
        super().__init__(**kwds)
        # record my name
        self._modelName = name
        # all done
        return


    def __str__(self):
        """
        Identify me by name
        """
        # easy enough
        return self._modelName


# end of file
