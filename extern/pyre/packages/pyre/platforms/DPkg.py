# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import re, subprocess
# framework
import pyre
# superclass
from .Managed import Managed


# declaration
class DPkg(Managed, family='pyre.platforms.packagers.dpkg'):
    """
    Support for the debian package manager
    """


    # constants
    name = 'dpkg'
    client = 'dpkg-query'
    defaultLocation = pyre.primitives.path('/usr/bin')


    # meta-methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # initialize the index of package alternatives for each category
        self._alternatives = {}
        # all done
        return


    # implementation details
    def alternatives(self, group):
        """
        Generate a sequence of alternative installations for {group}, starting with the default
        selection
        """
        # get the group category
        category = group.category
        # grab the index
        alternatives = self._alternatives
        # attempt to
        try:
            # look up the given {group} and pass on the package alternatives
            return alternatives[category]
        # if this fails
        except KeyError:
            # it's because we haven't encountered this group before; let's fix that
            pass

        # ask the package protocol to build one; this is done by the protocol since it knows
        # how to derive package names from version numbers correctly
        index = { name: package for name, package in group.dpkgAlternatives(dpkg=self) }
        # attach it
        alternatives[category] = index
        # and return it
        return index


    def identify(self, installation):
        """
        Attempt to map the package {installation} to the name of an installed package
        """
        # get the name of the installation instance
        name = installation.pyre_name
        # grab the index of installed packages
        installed = self.getInstalledPackages()

        # one possibility is that this instance belongs to a selection group; we build
        # these for some packages whenever we have to provide package choices to support the
        # computation of the package protocol default
        group = installation.pyre_implements
        # get the alternative index
        alternatives = self.alternatives(group=group)

        # beyond this point, nothing works unless this package belongs to a selection group
        if not alternatives:
            # it isn't
            msg = 'could not locate support for {!r} among {}'.format(name, alternatives)
            # complain
            raise installation.ConfigurationError(configurable=self, errors=[msg])

        # check whether the installation name is known
        try:
            # return the name of the package
            return alternatives[name]
        # if not
        except KeyError:
            # moving on
            pass

        # another approach is to attempt to find a selection that is related to the package
        # flavor; let's check
        try:
            # whether the installation has a flavor
            flavor = installation.flavor
        # if it doesn't
        except AttributeError:
            # describe what went wrong
            msg = "could not find a package installation for {!r}".format(name)
            # and report it
            raise package.ConfigurationError(configurable=self, errors=[msg])

        # collect all alternatives whose names start with the flavor
        candidates = [ tag for tag in alternatives if tag.startswith(flavor) ]

        # if there is exactly one candidate
        if len(candidates) == 1:
            # it's our best bet
            candidate = candidates[0]
            # find out which package implements it and return it
            return alternatives[candidate]

        # if there were no viable candidates
        if not candidates:
            # describe what went wrong
            msg = "no viable candidates for {!r}".format(flavor)
            # and report it
            raise installation.ConfigurationError(configurable=self, errors=[msg])

        # otherwise, there were more than one candidate; describe what went wrong
        msg = 'multiple candidates for {!r}: {}; please select one'.format(flavor, candidates)
        # and report it
        raise installation.ConfigurationError(configurable=self, errors=[msg])


    def setAlternatives(self, group, options):
        """
        Attach the table of available {options} for the given package {group}

        The table of options is a map from the constructed pyre legal installation names to the
        names of the packages that support them
        """
        # easy enough
        self._alternatives[group] = options
        # all done
        return options


    def getInstalledPackages(self):
        """
        Return the version and revision of all installed packages
        """
        # grab the index
        installed = self._installed
        # if it has not been initialized
        if installed is None:
            # prime it
            installed = {
                package: (version, revision)
                for package, version, revision in self.retrieveInstalledPackages()
            }
            # attach it
            self._installed = installed
        # ask it
        return installed


    def retrieveInstalledPackages(self):
        """
        Generate a sequence of all installed ports
        """
        # set up the shell command
        settings = {
            'executable': str(self.client),
            'args': (
                str(self.client),
                '--show',
                '--showformat=${binary:Package}\t${Version}\t${db:Status-Abbrev}\n',
            ),
            'stdout': subprocess.PIPE, 'stderr': subprocess.PIPE,
            'universal_newlines': True,
            'shell': False
        }
        # make a pipe
        with subprocess.Popen(**settings) as pipe:
            # get the text source
            stream = pipe.stdout
            # and info parser
            info = self._infoParser
            # grab the rest
            for line in pipe.stdout.readlines():
                # parse
                match = info.match(line)
                # if it matched
                if match:
                    # extract the information we need and hand it to the caller
                    yield match.group('package', 'version', 'revision')
        # all done
        return


    def retrievePackageContents(self, package):
        """
        Generate a sequence of the contents of {package}
        """
        # set up the shell command
        settings = {
            'executable': str(self.client),
            'args': (str(self.client), '--listfiles', package),
            'stdout': subprocess.PIPE, 'stderr': subprocess.PIPE,
            'universal_newlines': True,
            'shell': False
        }
        # execute
        with subprocess.Popen(**settings) as pipe:
            # get the text source
            stream = pipe.stdout
            # grab the rest
            for line in pipe.stdout.readlines():
                # strip it and hand it to the caller
                yield line.strip()
        # all done
        return


    # private data
    _installed = None
    _alternatives = None

    _infoParser = re.compile(
        r"(?P<package>[^\t:]+)(?P<arch>[^\t]+)?"
        r"\t"
        r"((?P<epoch>[\d]+):)?"
        r"(?P<version>[\w.+]+)"
        r"((?P<revision>[\w.+~-]+))?"
        r"\tii"
        )


# end of file
