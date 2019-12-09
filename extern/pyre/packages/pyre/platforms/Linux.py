# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import collections
import operator
import subprocess
# superclass
from .POSIX import POSIX
# the cpu info object
from .CPUInfo import CPUInfo


# declaration
class Linux(POSIX, family='pyre.platforms.linux'):
    """
    Encapsulation of a generic linux host
    """


    # public data
    platform = 'linux'
    distribution = 'generic'

    prefix_library = 'lib'
    extension_staticLibrary = '.a'
    extension_dynamicLibrary = '.so'

    template_staticLibrary = "{0.prefix_library}{1}{0.extension_staticLibrary}"
    template_dynamicLibrary = "{0.prefix_library}{1}{0.extension_dynamicLibrary}"


    # protocol obligations
    @classmethod
    def flavor(cls):
        """
        Return a suitable default encapsulation of the runtime host
        """
        # get the platform package
        import platform
        # identify the platform characteristics; careful not to set the {distribution}
        # attribute here; the subclasses set the distribution name to the pyre canonical
        # nickname
        distribution, cls.release, cls.codename = platform.linux_distribution()

        # check for ubuntu
        if distribution.lower().startswith('ubuntu'):
            # load the platform file
            from .Ubuntu import Ubuntu
            # and return it
            return Ubuntu
        # check for debian
        if distribution.lower().startswith('debian'):
            # load the platform file
            from .Debian import Debian
            # and return it
            return Debian
        # check for red hat
        if distribution.lower().startswith('red hat'):
            # load the platform file
            from .RedHat import RedHat
            # and return it
            return RedHat
        # check for centos
        if distribution.lower().startswith('centos'):
            # load the platform file
            from .CentOS import CentOS
            # and return it
            return CentOS

        # otherwise, act like a generic linux system
        return cls


    # implementation details: explorers
    @classmethod
    def cpuSurvey(cls):
        """
        Collect information about the CPU resources on this host
        """
        # first, let's try
        try:
            # to use {lscpu} to collect the information and return it
            return cls.lscpu()
        # if it's not available on this machine
        except FileNotFoundError:
            # no worries, we'll try something else
            pass

        # last resort, because it's heavily polluted by x86_64 peculiarities
        return cls.procCPUInfo()


    # implementation details: workhorses
    @classmethod
    def lscpu(cls):
        """
        Invoke {lscpu} to gather CPU info
        """
        # the name of the program that collates the cpu information
        client = 'lscpu'
        # the command line arguments
        settings = {
            'executable' : client,
            'args': (
                client,
                ),
            'stdout': subprocess.PIPE, 'stderr': subprocess.PIPE,
            'universal_newlines': True,
            'shell': False
        }

        # initialize storage
        sockets = 1
        coresPerSocket = 1
        threadsPerCore = 1

        # make a pipe
        with subprocess.Popen(**settings) as pipe:
            # get the text source and tokenize it
            tokens = cls.tokenizeCPUInfo(cpuinfo=pipe.stdout)
            # parse
            for key, value in tokens:
                # number of sockets
                if key == "Socket(s)":
                    # save
                    sockets = int(value)
                # number of cores per socket
                elif key == "Core(s) per socket":
                    # save
                    coresPerSocket = int(value)
                # number of threads per core
                elif key == "Thread(s) per core":
                    # save
                    threadsPerCore = int(value)

        # make a cpu info object
        info = CPUInfo()
        # decorate
        info.sockets = sockets
        info.cores = sockets * coresPerSocket
        info.cpus = info.cores * threadsPerCore
        # and retur it
        return info


    @classmethod
    def procCPUInfo(cls):
        """
        Interrogate /proc for CPU info

        This was the original manner in which pyre discovered cpu information. It appears that
        the gathering of information was inadvertently polluted by what is available for
        {x86_64} architectures, and fails to be useful on {ppc64le}. As a result, it has been
        replaced by the method {lscpu} above that seems to slower but much more reliable.
        """
        # initialize the cpu store
        ids = 0
        cpus = collections.defaultdict(dict)
        # the markers
        physicalid = None
        # prime the tokenizer
        tokens = cls.tokenizeCPUInfo(cpuinfo=open(cls.cpuinfo))
        # the keys we care about
        targets = {'siblings', 'cpu cores'}
        # parse
        for key, value in tokens:
            # if the key is blank
            if not key:
                # reset the marker
                physicalid = None
                # and move on
                continue
            # record the processor ids; that's all we have on single core machines
            if key == 'processor':
                # increment the count
                ids += 1
                # move on
                continue
            # the socket to which this core belongs
            if key == 'physical id':
                # harvest the cpu physical id
                physicalid = value
                # move on
                continue
            # harvest the interesting info
            if physicalid and key in targets:
                # attach it to the right socket
                cpus[physicalid][key] = value
                # and move on
                continue

        # initialize the counters
        sockets = physical = logical = 0
        # reduce
        for sec in cpus.values():
            # update the cpu count
            sockets += 1
            # update the number of physical cores
            physical += int(sec['cpu cores'])
            # update the number of logical cores
            logical += int(sec['siblings'])

        # create an info object
        info = CPUInfo()
        # if the reduction produced non-zero results
        if physical and logical:
            # decorate it
            info.sockets = sockets
            info.cores = physical
            info.cpus = logical
        # and return it
        return info


    @classmethod
    def tokenizeCPUInfo(cls, cpuinfo):
        """
        Split the CPU info file into (key, value) pairs
        """
        # in order to tokenize each line
        for line in cpuinfo:
            # strip whitespace
            line = line.strip()
            # if this leaves us with nothing, we ran into a separator blank line
            if not line:
                # form a pair of blank tokens
                key = value = ''
            # otherwise
            else:
                # split apart and strip leading and trailing whitespace
                key, value = map(operator.methodcaller('strip'), line.split(':'))
            # yield the tokens
            yield key, value
        # nothing more
        return


    # implementation constants
    issue = '/etc/issue'
    cpuinfo = '/proc/cpuinfo'


# end of file
