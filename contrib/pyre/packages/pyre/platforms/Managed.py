# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import re
# the framework
import pyre
# my protocol
from .PackageManager import PackageManager


# declaration
class Managed(pyre.component, implements=PackageManager):
    """
    Support for un*x systems that don't have package management facilities
    """


    # public data
    @property
    def name(self):
        """
        Get the name of this package manager
        """
        # the base class doesn't have one; subclasses must provide a unique name that enables
        # package categories to identify with which package manager they are collaborating
        raise NotImplementedError("class {.__name__} must supply a 'name'".format(type(self)))


    @property
    def client(self):
        """
        Get the name of the front end to the package manager database
        """
        # the base class doesn't have one; subclasses must provide the name or path to the
        # front end for their package manager

        # the error message template
        msg = (
            "class {.__name__} must supply 'client', "
            "the name or path to the package manager front end")
        # instantiate and complain
        raise NotImplementedError(msg.format(type(self)))


    # protocol obligations
    @pyre.export
    def prefix(self):
        """
        Retrieve the package manager install location
        """
        # check my cache
        prefix = self._prefix
        # for whether I have done this before
        if prefix is not None:
            # in which case I'm done
            return prefix
        # otherwise, grab the shell utilities
        import shutil
        # locate the full path to the package manager client
        client = shutil.which(self.client)
        # if we found it
        if client:
            # pathify
            client = pyre.primitives.path(client)
        # otherwise
        else:
            # maybe it's not on the path; try the default
            client = self.defaultLocation / self.client
            # if it's not there
            if not client.exists():
                # build the message
                msg = 'could not locate {.manager!r}'.format(self)
                # complain
                raise self.ConfigurationError(configurable=self, errors=[msg])

        # found it; let's remember its exact location
        self.client = client
        # extract the parent directory
        bin = client.parent
        # and once again to get the prefix
        prefix = bin.parent
        # remember it for next time
        self._prefix = prefix
        # and return it
        return prefix


    @pyre.export
    def installed(self):
        """
        Retrieve available information for all installed packages
        """
        # ask the index...
        return self.getInstalledPackages()


    @pyre.export
    def info(self, package):
        """
        Return the available information about {package}
        """
        # send what the index has
        return self.getInstalledPackages()[package]


    @pyre.provides
    def packages(self, category):
        """
        Provide a sequence of package names that provide compatible installations for the given
        package {category}.
        """
        # check whether this package category can interact with me
        try:
            # by looking for my handler
            choices = getattr(category, '{}Packages'.format(self.name))
        # if it can't
        except AttributeError:
            # the error message template
            template = "the package category {.category!r} does not support {.name!r}"
            # build the message
            msg = template.format(category, self)
            # complain
            raise self.ConfigurationError(configurable=category, errors=[msg])

        # otherwise, ask the package category to do dpkg specific hunting
        yield from choices(packager=self)

        # all done
        return


    @pyre.export
    def contents(self, package):
        """
        Retrieve the contents of the {package}
        """
        # ask port for the package contents
        yield from self.retrievePackageContents(package=package)
        # all done
        return


    @pyre.provides
    def configure(self, installation):
        """
        Dispatch to the {installation} configuration procedure that is specific to this package
        manager
        """
        # check whether this installation can interact with me
        try:
            # by looking for my handler
            configure = getattr(installation, self.name)
        # if it can't
        except AttributeError:
            # the error message template
            template = "the package flavor {.flavor!r} does not support {.name!r}"
            # build the message
            msg = template.format(installation, self)
            # complain
            raise self.ConfigurationError(configurable=installation, errors=[msg])

        # otherwise, ask the installation to configure itself with my help
        return configure(packager=self)


    # implementation details
    def find(self, target, pile):
        """
        Interpret {target} as a regular expression and return a sequence of the contents of {pile}
        that match it.

        This is intended as a way to scan through the contents of packages to find a path that
        matches {target}
        """
        # compile the target regex
        regex = re.compile(target)

        # go through the pile
        for item in pile:
            # check
            match = regex.match(item)
            # if it matches
            if match:
                # hand it to the caller
                yield match

        # all done
        return


    def findfirst(self, target, contents):
        """
        Locate the path to {target} in the {contents} of some package
        """
        # form the regex
        regex = '(?P<path>.*)/{}$'.format(target)
        # search for it in contents
        for match in self.find(target=regex, pile=contents):
            # extract the folder
            return pyre.primitives.path(match.group('path'))
        # otherwise, leave it blank
        return


    def locate(self, targets, paths):
        """
        Generate a sequence of the full {paths} to the {targets}
        """
        # go through the targets
        for target in targets:
            # and each of paths
            for path in paths:
                # form the combination
                candidate = path / target
                # check whether it exists
                if candidate.exists():
                    # got one
                    yield candidate
                    # grab the next
                    break
        # all done
        return


    # private data
    # the installation location of the package manager
    _prefix = None


# end of file
