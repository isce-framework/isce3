# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# the framework
import pyre
# superclass
from .POSIX import POSIX
# the default package manager
from .MacPorts import MacPorts
# the CPU info object
from .CPUInfo import CPUInfo


# declaration
class Darwin(POSIX, family='pyre.platforms.darwin'):
    """
    Encapsulation of a darwin host
    """


    # constants
    platform = 'darwin'
    distribution = 'osx'

    prefix_library = 'lib'
    extension_staticLibrary = '.a'
    extension_dynamicLibrary = '.dylib'

    template_staticLibrary = "{0.prefix_library}{1}{0.extension_staticLibrary}"
    template_dynamicLibrary = "{0.prefix_library}{1}{0.extension_dynamicLibrary}"

    # user configurable state
    packager = pyre.platforms.packager(default=MacPorts)
    packager.doc = 'the manager of external packages installed on this host'


    # implementation details: explorers
    @classmethod
    def cpuSurvey(cls):
        """
        Collect information about the CPU resources on this host
        """
        # on darwin, we get this information from an extension
        try:
            # we get this information from an extension
            from pyre.extensions import host
        # which may not be accessible, e.g. if pyre is executed from within a zipfile
        except ImportError:
            # revert to defaults
            return super().cpuSurvey()
        # otherwise, initialize the info object
        info = CPUInfo()
        # decorate it; macos doesn't seem to know the number of CPU sockets on the motherboard
        info.cores = host.physicalMax()
        info.cpus = host.logicalMax()
        # and return it
        return info


    @classmethod
    def getOSInfo(cls):
        # ask the platform package
        import platform
        # for the release number
        release, _, _ = platform.mac_ver()
        # extract the major release
        major = '.'.join(release.split('.')[:2])
        # use it to get the codename
        codename = cls.codenames.get(major, 'unknown')
        # and return
        return release, codename


    # meta-methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # deduce the OS release and codename
        self.release, self.codename = self.getOSInfo()
        # all done
        return


    # private data
    # the known code names
    codenames = {
        '10.14': 'mojave',
        '10.13': 'high sierra',
        '10.12': 'sierra',
        '10.11': 'el capitan',
        '10.10': 'yosemite',
        '10.9': 'mavericks',
        '10.8': 'mountain lion',
        '10.7': 'lion',
        '10.6': 'snow leopard',
        '10.5': 'leopard',
        '10.4': 'tiger',
        '10.3': 'panther',
        '10.2': 'jaguar',
        '10.1': 'puma',
        '10.0': 'cheetah',
        }


# end of file
