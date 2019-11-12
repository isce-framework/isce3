# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import os
# framework
import pyre


# the base manager of specific package installations
class Installation(pyre.component):
    """
    Base class for all package installations
    """

    # constants
    category = 'unknown'

    # public state
    version = pyre.properties.str(default="unknown")
    version.doc = 'the package version'

    prefix = pyre.properties.path()
    prefix.doc = 'the package installation directory'


    # public data
    @property
    def majorver(self):
        """
        Extract the portion of a version number that is used to label my parts
        """
        # get my version
        version = self.version
        # attempt to
        try:
            # split my version into major, minor and the rest
            major, *rest = version.split('.')
        # if i don't have enough fields
        except ValueError:
            # can't do much
            return version
        # otherwise, assemble the significant part and return it
        return major


    @property
    def sigver(self):
        """
        Extract the portion of a version number that is used to label my parts
        """
        # get my version
        version = self.version
        # attempt to
        try:
            # split my version into major, minor and the rest
            major, minor, *rest = version.split('.')
        # if i don't have enough fields
        except ValueError:
            # can't do much
            return version
        # otherwise, assemble the significant part and return it
        return '{}.{}'.format(major, minor)


    # framework hooks
    def pyre_configured(self):
        """
        Verify the package configuration
        """
        # chain up
        yield from super().pyre_configured()
        # if i don't have a good version
        if self.version == 'unknown':
            # complain
            yield 'unknown version'

        # get my prefix
        prefix = self.prefix
        # if it's empty
        if not prefix:
            # complain
            yield "empty prefix"
        # if not but set to something that's not a directory
        elif not prefix.isDirectory():
            # mark as bad attempt to configure
            self._misconfigured = True
            # complain
            yield "invalid prefix '{}'".format(prefix)

        # all done
        return


    def pyre_initialized(self):
        """
        Attempt to repair broken configurations
        """
        # grab my configuration errors
        if not self.pyre_configurationErrors:
            # if there weren't any, we are done
            return
        # if the configuration errors were caused by the user
        if self._misconfigured:
            # don't try to repair the user's mess since there is no way of knowing what to do
            yield from self.pyre_configurationErrors
            # indicate we are giving up
            yield "automatic configuration aborted"
            # and do nothing else
            return

        # otherwise, we have work to do; grab the package manager
        packager = self.pyre_host.packager
        # and attempt to
        try:
            # get him to help me repair this configuration
            packager.configure(installation=self)
        # if something went wrong
        except self.ConfigurationError as error:
            # report my errors
            yield from error.errors

        # all done
        return


    # configuration validation
    def verify(self, trait, patterns=(), folders=()):
        """
        Verify that {trait} properly configured by checking that every file name in {patterns}
        exists in one of the {folders}
        """
        # if the list of {folders} is empty
        if not folders:
            # complain
            yield "empty {}".format(trait)
            # and stop
            return
        # put all the folders in a set
        good = set(folders)
        # check that all the {folders}
        for folder in folders:
            # are valid
            if not folder.isDirectory():
                # if not, mark this as a bad attempt to configure
                self._misconfigured = True
                #  complain
                yield "'{}' is not a valid directory".format(folder)
                # and remove it from the good pile
                good.remove(folder)

        # go through the list of filenames
        for pattern in patterns:
            # check whether each of the good folders
            for folder in good:
                # contains
                try:
                    # files that match
                    next(folder.glob(pattern))
                # if not
                except StopIteration:
                    # move on
                    continue
                # if it's there
                break
            # if we couldn't locate this file
            else:
                # mark as bad attempt to configure
                self._misconfigured = True
                # complain
                yield "couldn't locate {!r}".format(pattern)

        # all done
        return


    def commonpath(self, folders):
        """
        Find the longest prefix common to the given {folders}
        """
        # convert the paths into a sequence of strings
        folders = tuple(map(str, folders))
        # compute and return the longest common prefix
        return os.path.commonpath(folders)


    def join(self, folders, prefix=''):
        """
        Render the sequence of {folders} as a flat string with each one prefixed by {prefix}
        """
        # splice it all together and return it
        return " ".join("{}{}".format(prefix, folder) for folder in folders)


    # private data
    _misconfigured = False


# end of file
