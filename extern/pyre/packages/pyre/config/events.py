# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
The various codecs build and populate instances of these classes as part of the ingestion of
configuration sources. They provide a temporary holding place to store the harvested events
until the entire source is processed without errors. This way, the configuration retrieved from
the source will be known to be at least syntactically correct without the risk of polluting the
global framework configuration data structures with partial input from invalid sources.
"""


class Event:
    """The base class for all configuration events"""

    # public data
    locator = None

    # meta methods
    def __init__(self, locator, **kwds):
        super().__init__(**kwds)
        self.locator = locator
        return


class Command(Event):
    """A command"""

    # public data
    command = None

    # interface
    def identify(self, inspector, **kwds):
        """Ask {inspector} to process a {Command}"""
        return inspector.execute(command=self, **kwds)

    # meta methods
    def __init__(self, command, **kwds):
        super().__init__(**kwds)
        self.priority = None
        self.command = command
        return

    def __str__(self):
        return "{{{}: {}}}".format(self.locator, self.command)


class Assignment(Event):
    """A request to bind a {key} to a {value}"""

    # public data
    key = None
    value = None

    # interface
    def identify(self, inspector, **kwds):
        """Ask {inspector} to process an {Assignment}"""
        return inspector.assign(assignment=self, **kwds)

    # meta methods
    def __init__(self, key, value, **kwds):
        super().__init__(**kwds)
        self.key = key
        self.value = value
        return

    def __str__(self):
        return "{{{}: {} <- {}}}".format(self.locator, self.key, self.value)


class ConditionalAssignment(Assignment):
    """A request to bind a {key} to a {value} subject to a condition"""

    # public data
    component = None
    conditions = None

    # interface
    def identify(self, inspector, **kwds):
        """Ask {inspector} to process a {ConditionalAssignment}"""
        return inspector.defer(assignment=self, **kwds)

    # meta methods
    def __init__(self, component, condition, **kwds):
        super().__init__(**kwds)
        self.component = component
        self.conditions = [condition]
        return

    def __str__(self):
        msg = [
            "{{{.locator}:".format(self),
            "  {0.component}: {0.key} <- {0.value!r}".format(self),
            "  subject to:"
            ]
        for name, family in self.conditions:
            msg.append("    name={}, family={}".format(name, family))
        msg.append("}")

        return "\n".join(msg)


class Source(Event):
    """A request to load configuration settings from a named source"""

    # public data
    source = None

    # interface
    def identify(self, inspector, **kwds):
        """Ask {inspector} to process a {Source} event"""
        return inspector.load(request=self, **kwds)

    # meta methods
    def __init__(self, source, **kwds):
        super().__init__(**kwds)
        self.source = source
        return

    def __str__(self):
        return "{{{0.locator}: loading {0.source}}".format(self)


# end of file
