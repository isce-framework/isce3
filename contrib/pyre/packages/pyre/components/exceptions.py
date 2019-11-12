# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Definitions for all the exceptions raised by this package
"""


from ..framework.exceptions import FrameworkError


class ComponentError(FrameworkError):
    """
    Base class for component specification errors
    """


class CategoryMismatchError(ComponentError):
    """
    Exception raised when two configurables have traits by the same name but have different
    categories
    """

    # public data
    description = "category mismatch in trait {0.name!r} between {0.configurable} and {0.target}"

    # meta-methods
    def __init__(self, configurable, target, name, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the error info
        self.configurable = configurable
        self.target = target
        self.name = name
        # all done
        return


class ImplementationSpecificationError(ComponentError):
    """
    Exception raised when the {implements} specification of a component declaration contains
    errors, e.g. classes that don't derive from Protocol
    """

    # public data
    description = '{0.name}: poorly formed implementation specification'

    # meta-methods
    def __init__(self, name, errors, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the error info
        self.name = name
        self.errors = errors
        # all done
        return


class ProtocolError(ComponentError):
    """
    Exception raised when a component does not implement correctly the protocols in its
    implementation specification
    """

    # meta-methods
    def __init__(self, component, protocol, report, **kwds):
        # chain up
        super().__init__(**kwds)
        # and record the error conditions for whomever may be catching this exception
        self.component = component
        self.protocol = protocol
        self.report = report
        # extract the actual protocols, skipping {object}
        protocols = tuple(str(base) for base in protocol.pyre_pedigree)
        # support for singular/plural
        s = '' if len(protocols) == 1 else 's'
        # here is the error description
        self.description = (
            "{{0.component}} does not implement correctly the following protocol{}: {}"
            .format(s, ", ".join(protocols)))

        # all done
        return


class TraitNotFoundError(ComponentError):
    """
    Exception raised when a request for a trait fails
    """

    # public data
    description = "{0.configurable} has no trait named {0.name!r}"

    # meta-methods
    def __init__(self, configurable, name, **kwds):
        # pass it on
        super().__init__(**kwds)
        # save the source of the error
        self.configurable = configurable
        self.name = name
        # all done
        return


class FacilitySpecificationError(ComponentError):
    """
    Exception raised when a facility cannot instantiate its configuration specification
    """

    # public data
    description = "{0.__name__}.{0.trait.name}: could not instantiate {0.value!r}"

    # meta-methods
    def __init__(self, configurable, trait, value, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the error info

        self.configurable = configurable
        self.trait = trait
        self.value = value
        # all done
        return


class ProtocolCompatibilityError(ComponentError):
    """
    Exception raised when a configurable is incompatible with a suggested protocol
    """

    # public data
    description = '{0.configurable} is incompatible with {0.protocol}'

    # meta-methods
    def __init__(self, configurable, protocol, **kwds):
        # chain up
        super().__init__(**kwds)
        # store my context
        self.configurable = configurable
        self.protocol = protocol
        # all done
        return


class ResolutionError(ComponentError):
    """
    Exception raised when a protocol cannot resolve a string into a component
    """

    # public data
    description = 'could not resolve {0.value!r} into a component that implements {0.protocol}'

    # meta-methods
    def __init__(self, protocol, value, **kwds):
        # chain up
        super().__init__(**kwds)
        # store my context
        self.protocol = protocol
        self.value = value
        # all done
        return


class DefaultError(ComponentError):
    """
    Exception raised when a protocol cannot determine a valid default value
    """

    # public data
    description = 'no valid default binding for {0.protocol}'

    # meta-methods
    def __init__(self, protocol, **kwds):
        # chain up
        super().__init__(**kwds)
        # store my context
        self.protocol = protocol
        # all done
        return


class ConfigurationError(ComponentError):
    """
    Exception raised when something bad happens during component configuration
    """

    # public data
    description = 'while configuring {0.configurable}:\n    {0.report}'

    @property
    def report(self):
        """
        Splice my errors together
        """
        return "\n    ".join(map(str, self.errors))

    # meta-methods
    def __init__(self, configurable, errors, **kwds):
        # chain up
        super().__init__(**kwds)
        # store my context
        self.configurable = configurable
        self.errors = errors
        # all done
        return


class InitializationError(ComponentError):
    """
    Exception raised when something bad happens during component initialization
    """

    # public data
    description = 'while initializing {0.configurable}:\n    {0.report}'

    @property
    def report(self):
        """
        Splice my errors together
        """
        return "\n    ".join(map(str, self.errors))

    # meta-methods
    def __init__(self, configurable, errors, **kwds):
        # chain up
        super().__init__(**kwds)
        # store my context
        self.configurable = configurable
        self.errors = errors
        # all done
        return


# end of file
