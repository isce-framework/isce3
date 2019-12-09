#!/usr/bin/env python3
# -*- Python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import os
# access the framework
import pyre
# my protocols
from .Project import Project


# the application class
class Smith(pyre.application, family='pyre.applications.smith'):
    """
    A generator of projects in pyre standard form
    """


    # user configurable state
    project = Project()
    project.doc = "the project information"


    # public data
    @property
    def vault(self):
        """
        Return the location of the project template directory
        """
        # build and  return the absolute path to the model template
        return pyre.prefix / 'templates' / self.project.template


    # application obligations
    @pyre.export
    def main(self, *args, **kwds):
        # get the name of the project
        project = self.project.name
        # instantiate my host configuration so that its settings materialize
        host = self.project.live
        # get the nameserver
        nameserver = self.pyre_nameserver
        # make local filesystem rooted at the model template directory
        template = self.vfs.local(root=self.vault).discover()

        # make a local filesystem rooted at the current directory
        cwd = self.vfs.local(root='.').discover()
        # if the target path exists already
        if project in cwd:
            # complain
            self.error.log("the folder {!r} exists already".format(project))
            # report failure
            return 1

        # show me
        self.info.log('building the bzr repository')
        # have {bazaar} create the directory
        os.system("bzr init -q {}".format(project))

        self.info.log('generating the source tree')
        # initialize the workload
        todo = [(cwd, project, template)]
        # as long as there are folders to visit
        for destination, name, source in todo:
            # show me
            # self.info.log('creating the folder {!r}'.format(name))
            # create the new folder
            folder = cwd.mkdir(parent=destination, name=name, exist_ok=True)
            # go through the folder contents
            for entry, child in source.contents.items():
                # attempt to
                try:
                    # expand any macros in the name
                    entry = nameserver.interpolate(expression=entry)
                # if anything goes wrong
                except self.FrameworkError as error:
                    # generate an error message
                    self.error.log('could not process {}: {}'.format(entry, error))
                    # and move on
                    continue
                # show me
                # self.info.log('generating {!r}'.format(entry))
                # if the {child} is a folder
                if child.isFolder:
                    # add it to the workload
                    todo.append((folder, entry, child))
                    # and move on
                    continue

                # if the name is blacklisted
                if self.project.blacklisted(filename=entry):
                    # open the file in binary mode and read its contents
                    body = child.open(mode='rb').read()
                    # and copy it
                    destination = cwd.write(parent=folder, name=entry, contents=body, mode='wb')
                # otherwise
                else:
                    # the {child} is a regular file; open it and read its contents
                    body = child.open().read()
                    # attempt to
                    try:
                        # expand any macros
                        body = nameserver.interpolate(expression=body)
                    # if anything goes wrong
                    except self.FrameworkError as error:
                        # generate an error message
                        self.error.log('while processing {}: {}'.format(entry, error))
                        # and move on
                        continue
                    # create the file
                    destination = cwd.write(parent=folder, name=entry, contents=body)

                # in any case, get me the meta data
                metaold = template.info(child)
                metanew = cwd.info(destination)
                # adjust the permissions of the new file
                metanew.chmod(metaold.permissions)

        # tell me
        self.info.log('committing the initial revision')
        # build the commit command
        command = (
            "unset CDPATH; " # just in case the user has strange tastes
            "cd {}; "
            "bzr add -q; "
            "bzr commit -q -m 'automatically generated source'; "
            "cd ..").format(project)
        # execute
        os.system(command)

        # return success
        return 0


# end of file
