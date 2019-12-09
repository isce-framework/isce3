# -*- Python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
merlin is a package intended as a replacement to {make}
"""


def main():
    """
    This is the main entry point in the package. It is invoked by the {merlin} script.  Its job
    is to boot pyre, examine the command line to deduce which actor the user would like to
    invoke, instantiate it, and call its main entry point with the supplied command line
    arguments.

    There are other possible ways to invoke merlin. See the package documentation.
    """
    # let the plexus do its thing
    return merlin.run()


# administrative
def credits():
    """
    Print the acknowledgments
    """
    # generate the message
    return print(meta.acknowledgments)


def copyright():
    """
    Return the merlin copyright note
    """
    # generate the message
    return print(meta.copyright)


def license():
    """
    Print the merlin license
    """
    # generate the message
    return print(meta.license)


def version():
    """
    Return the merlin version
    """
    # return the version
    return meta.version


# pull the meta-data
from . import meta

# bootstrapping
def boot():
    # check whether
    try:
        # the user
        import __main__
        # has indicated we should skip booting
        if __main__.merlin_noboot:
            # in which case, do not build a plexus
            return None
    # if anything goes wrong
    except:
        # just ignore it and carry on
        pass

    # package registration
    import pyre
    # register the package
    global package
    package = pyre.executive.registerPackage(name='merlin', file=__file__)
    # attach the geography
    global home, prefix, defaults
    home, prefix, defaults = package.layout()

    # externals
    import weakref
    # access the plexus factory
    from .components import merlin
    # build one and return it
    plexus = merlin(name='merlin.plexus')

    # get the dashboard
    from .components import dashboard
    # attach the singletons
    dashboard.merlin = weakref.proxy(plexus)

    # all done
    return plexus


# the framework entities
from pyre import foundry, export, properties, protocol
from .components import component, action, spell


# convenience
def error(message):
    """
    Generate an error message
    """
    # get the logging mechanism
    import journal
    # build an error message object in my namespace
    error = journal.error('merlin')
    # log and return
    return error.log(message)

def warning(message):
    """
    Generate a warning
    """
    # get the logging mechanism
    import journal
    # build a warning object in my namespace
    warning = journal.warning('merlin')
    # log and return
    return warning.log(message)

def info(message):
    """
    Generate an informational message
    """
    # get the logging mechanism
    import journal
    # build an informational message object in my namespace
    info = journal.info('merlin')
    # log and return
    return info.log(message)


# the package
package = None
# geography
# the directory of the package
home = None
# the pathname of the installation
prefix = None
# the directory with the system defaults
defaults = None
# the singleton
merlin = boot()

# end of file
