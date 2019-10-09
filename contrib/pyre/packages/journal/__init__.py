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
    return _journal_copyright


def license():
    """
    Print the pyre journal license
    """
    print(_journal_license)
    return


def version():
    """
    Return the pyre journal version
    """
    return _journal_version


# license
_journal_version = (1, 0, 0)

_journal_copyright = "pyre journal: Copyright (c) 1998-2019 Michael A.G. Aïvázis"

_journal_license = """
    pyre journal 1.0
    Copyright (c) 1998-2019 Michael A.G. Aïvázis
    All Rights Reserved


    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.

    * Neither the name pyre nor the names of its contributors may be
      used to endorse or promote products derived from this software
      without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
    COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
    ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    """


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
    Channel.journal = Journal()

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

# grab the framework
import pyre
# register the package
package = pyre.executive.registerPackage(name='journal', file=__file__)
# record the layout
home, prefix, defaults = package.layout()

# access to the public names
# the channel categories
from .Debug import Debug as debug
from .Firewall import Firewall as firewall
from .Info import Info as info
from .Warning import Warning as warning
from .Error import Error as error

# devices
from .Console import Console as console
from .File import File as file

# the package exception
from .exceptions import FirewallError

# make it so...
extension = boot()


# end of file
