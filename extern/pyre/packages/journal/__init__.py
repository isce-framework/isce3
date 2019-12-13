# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# administrative
def copyright():
    """
    Return the pyre journal copyright note
    """
    # easy enough
    return print(meta.header)


def license():
    """
    Print the pyre journal license
    """
    # easy enough
    return print(meta.license)


def version():
    """
    Return the pyre journal version
    """
    return meta.version


# the bootstrapping logic is tucked away in a function to prevent namespace pollution
def boot():
    """
    Initialize the journal package.

    Attempt to locate the C++ extension and use it if available; fall back on the pure python
    implementation. Either way, return a marker that enables clients to check whether there is
    support for journal messages from C/C++/FORTRAN.
    """
    # access to the local types
    from .Journal import Journal
    from .Channel import Channel
    # instantiate the journal component and patch {Channel}
    Channel.journal = Journal(name="pyre.journal")

    # attempt to load the journal extension
    try:
        from . import journal
    # if it fails for any reason
    except Exception:
        # ignore it; the default implementation will kick in
        extension = None
    # otherwise
    else:
        # save the extension module
        extension = journal
        # hand the journal instance to the extension module so it can have access to the
        # default device
        journal.registerJournal(Channel.journal)

        # attach the indices from the extension module to the channel categories
        # access the index factories
        from . import proxies
        # install
        debug._index = proxies.debugIndex()
        firewall._index = proxies.firewallIndex()
        info._index = proxies.infoIndex()
        warning._index = proxies.warningIndex()
        error._index = proxies.errorIndex()

    # transfer settings from the configuration store
    categories = [ debug, firewall, info, warning, error ]
    Channel.journal.configureCategories(categories)

    # all done
    return extension


# access to the singleton
def scribe():
    """
    Provide access to the journal manager
    """
    # channel knows...
    from .Channel import Channel as channel
    # so make him say
    return channel.journal


# administrative
from . import meta
# grab the framework
import pyre
# register the package
package = pyre.executive.registerPackage(name='journal', file=__file__)
# record the layout
home, prefix, defaults = package.layout()

# access to the public names
# the channel factories
from .Debug import Debug as debug
from .Firewall import Firewall as firewall
from .Info import Info as info
from .Warning import Warning as warning
from .Error import Error as error

# the protocols
from . import protocols


# device foundries
@pyre.foundry(implements=protocols.device)
def console():
    # grab the implementation
    from .Console import Console
    # steal its docstrnig
    __doc__ = Console.__doc__
    # and publish it
    return Console

@pyre.foundry(implements=protocols.device)
def file():
    # grab the implementation
    from .File import File
    # steal its docstrnig
    __doc__ = File.__doc__
    # and publish it
    return File


# the package exception
from .exceptions import FirewallError


# make it so...
extension = boot()


# end of file
