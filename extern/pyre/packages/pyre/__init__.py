# -*- Python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
pyre is a framework for building flexible applications

For more details, see http://pyre.orthologue.com.
For terms of use, see pyre.license()
"""

# version check
# python version information is in {sys.version}
import sys
# unwrap
major, minor, _, _, _ = sys.version_info
# check
if major < 3 or (major == 3 and minor < 6):
    # complain
    raise RuntimeError("pyre needs python 3.6 or newer")


# convenience
def resolve(uri):
    """
    Interpret {uri} as a request to locate and load a component
    """
    # build a locator
    caller = tracking.here(level=1)
    # get the executive to retrieve candidates
    for component in executive.resolve(uri=uri):
        # adjust its locator
        component.pyre_locator = tracking.chain(caller, component.pyre_locator)
        # and return it
        return component
    # otherwise, the {uri} could not be resolved
    return


def loadConfiguration(uri):
    """
    Open {uri} and attempt to load its contents into the configaration model
    """
    # build a locator for these settings
    here = tracking.here(level=1)
    # get the executive to do its thing
    return executive.loadConfiguration(uri=uri, locator=here)


# version dependent constants
def computeCallerStackDepth():
    """
    Compute the stack depth offset to get to the caller of a function
    """
    # computed in {tracking}
    return tracking.callerStackDepth


# administrative
def copyright():
    """
    Return the pyre copyright note
    """
    return print(meta.header)


def license():
    """
    Print the pyre license
    """
    # print it
    return print(meta.license)


def version():
    """
    Return the pyre version
    """
    return meta.version


def credits():
    """
    Print the acknowledgments
    """
    # print it
    return print(meta.acknowledgments)


# component introspection
def where(configurable, attribute=None):
    """
    Retrieve the location where the {attribute} of {configurable} got its value; if no
    {attribute} is specified, retrieve information about the {configurable} itself
    """
    # if no attribute name is given, return the locator of the configurable
    if attribute is None: return configurable.pyre_locator
    # retrieve the trait descriptor
    trait = configurable.pyre_trait(alias=attribute)
    # find the slot where the attribute is stored
    slot = configurable.pyre_inventory[trait]
    # and return its locator
    return slot.locator


# put the following start-up steps inside functions so we can have better control over their
# execution context and namespace pollution
def boot():
    """
    Perform all the initialization steps necessary to bootstrap the framework
    """
    # check the version of python
    import sys
    major, minor, micro, _, _ = sys.version_info
    if major < 3 or (major == 3 and minor < 2):
        from .framework.exceptions import PyreError
        raise PyreError(description="pyre needs python 3.2 or newer")

    # check whether the user has indicated we should skip booting
    try:
        import __main__
        if __main__.pyre_noboot: return None
    # if anything goes wrong
    except:
        # just ignore it and carry on
        pass

    # grab the executive factory
    from . import framework
    # build one and return it
    return framework.executive().boot()


def debug():
    """
    Enable debugging of pyre modules.

    Modules that support debugging must provide a {debug} method and do as little as possible
    during their initialization. The fundamental constraints are provided by the python import
    algorithm that only give you one chance to import a module.

    This must be done very early, before pyre itself starts importing its packages. One way to
    request debugging is to create a variable {pyre_debug} in the __main__ module that contains
    a set of strings, each one of which is the name of a pyre module that you would like to
    debug.
    """
    # the set of packages to patch for debug support
    packages = set()
    # pull pyre_debug from the __main__ module
    import __main__
    try:
        packages |= set(__main__.pyre_debug)
    except:
        pass
    # iterate over the names, import the package and invoke its debug method
    for package in packages:
        module = __import__(package, fromlist=["debug"])
        module.debug()
    # all done
    return


# kickstart
# invoke the debug method in case the user asked for debugging support
debug()

# version info
from . import meta
# convenient access to parts of the framework
from . import version, constraints, geometry, primitives, tracking
# configurables and their support
from .components.Actor import Actor as actor
from .components.Role import Role as role
from .components.Protocol import Protocol as protocol
from .components.Component import Component as component
from .components.Foundry import Foundry as foundry
from .components.Monitor import Monitor as monitor
from .components.Tracker import Tracker as tracker
# traits
from .traits import properties
property = properties.identity
from .traits.Behavior import Behavior as export
from .traits.Behavior import Behavior as provides
from .traits.Facility import Facility as facility

# the base class of all pyre exceptions
from .framework.exceptions import PyreError

# build the executive
executive = boot()
# if the framework booted properly
if executive:
    # turn on the executive
    executive.activate()
    # register this package
    package = executive.registerPackage(name='pyre', file=__file__)
    # record its geography
    home, prefix, defaults = package.layout()
    # package managers
    from . import externals
    # platform managers
    from . import platforms
    # discover information about the runtime environment
    executive.discover()
    # application shells
    from .shells import application, action, plexus, command, panel
    # support for filesystems
    from . import filesystem
    # support for workflows
    from . import flow
    # document rendering
    from . import weaver
    # the interprocess communication mechanisms
    from . import ipc, nexus, services


# clean up the executive instance when the interpreter shuts down
import atexit
@atexit.register
def shutdown():
    """
    Attempt to hunt down and destroy all known references to the executive
    """
    # access the executive
    global executive
    # if there is one
    if executive:
        # ask it to clean up after itself
        executive.shutdown()
        # zero out the global reference
        executive = None
    # that should be enough
    return


# end of file
