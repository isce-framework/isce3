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
    """
    The base class for all configuration events
    """

    # public data
    locator = None

    # meta methods
    def __init__(self, locator, **kwds):
        # chain up
        super().__init__(**kwds)
        # record the locator
        self.locator = locator
        # all done
        return


class Command(Event):
    """
    A command
    """

    # public data
    command = None

    # interface
    def identify(self, inspector, **kwds):
        """Ask {inspector} to process a {Command}"""
        # forward to the {inspector}
        return inspector.execute(command=self, **kwds)

    # meta methods
    def __init__(self, command, **kwds):
        # chain up
        super().__init__(**kwds)
        # record the priority
        self.priority = None
        # and the command itself
        self.command = command
        # all done
        return

    def __str__(self):
        return f"{{{self.locator}: {self.command}}}"


class Assignment(Event):
    """
    A request to bind a {key} to a {value}
    """

    # public data
    key = None
    value = None

    # interface
    def identify(self, inspector, **kwds):
        """Ask {inspector} to process an {Assignment}"""
        # forward to the {inspector}
        return inspector.assign(assignment=self, **kwds)

    # meta methods
    def __init__(self, key, value, **kwds):
        # chain up
        super().__init__(**kwds)
        # record the key
        self.key = key
        # and the value
        self.value = value
        # all done
        return

    def __str__(self):
        return f"{{{self.locator}: {self.key} <- {self.value}}}"


class ConditionalAssignment(Assignment):
    """
    A request to bind a {key} to a {value} subject to a condition
    """

    # public data
    component = None
    conditions = None

    # interface
    def identify(self, inspector, **kwds):
        """Ask {inspector} to process a {ConditionalAssignment}"""
        # forward to the {inspector}
        return inspector.defer(assignment=self, **kwds)

    # meta methods
    def __init__(self, component, condition, **kwds):
        # chain up
        super().__init__(**kwds)
        # record the component
        self.component = component
        # and the conditions under which this assignment is to be applied
        self.conditions = [condition]
        # all done
        return

    def __str__(self):
        # initialize the rendering
        msg = [
            f"{{{self.locator}:",
            f"  {self.component}: {self.key} <- {self.value!r}",
            "  subject to:"
            ]
        # go through the conditions
        for name, family in self.conditions:
            # and add them to the pile
            msg.append(f"    name={name}, family={family}")
        # close the delimiter
        msg.append("}")
        # assemble and return
        return "\n".join(msg)


class Source(Event):
    """
    A request to load configuration settings from a named source
    """

    # public data
    source = None

    # interface
    def identify(self, inspector, **kwds):
        """Ask {inspector} to process a {Source} event"""
        # forward to the {inspector}
        return inspector.load(request=self, **kwds)

    # meta methods
    def __init__(self, source, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the configuration source
        self.source = source
        # all done
        return

    def __str__(self):
        return "{{{self.locator}: loading {self.source}}"


# end of file
