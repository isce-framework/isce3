# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import os
import merlin


# declaration
class Initializer(merlin.spell):
    """
    Create a new merlin project rooted at the given directory
    """


    # public state
    project = merlin.properties.str(default=None)
    project.doc = 'the name of the project'

    createPrefix = merlin.properties.bool(default=False)
    createPrefix.aliases.add('create-prefix')
    createPrefix.doc = 'create all directories leading up to the specified target'

    force = merlin.properties.bool(default=False)
    force.doc = 'initialize the target folder regardless of whether is it already part of a project'


    # class interface
    @merlin.export
    def main(self, plexus, argv):
        """
        Make {folder} the root of a new merlin project. The target {folder} is given as an
        optional command line argument, and defaults to the current directory. Issue an error
        message if {folder} is already a merlin project.
        """
        # NYI: non-local uris
        # access my executive
        merlin = self.merlin

        # extract my arguments
        folders = list(argv) or [os.curdir]
        # if there is more than one
        if len(folders) > 1:
            # issue a warning
            plexus.warning.log(
                'cannot initialize multiple project folders; ignoring all but the first')
        # extract the folder
        folder = folders[0]

        # first check whether this directory is already part of a merlin project
        root, metadir = merlin.locateProjectRoot(folder=folder)
        # if it is
        if root and not self.force:
            # complain
            return plexus.error.log('{!r} is already within an existing project'.format(folder))

        # if the directory does not exist
        if not os.path.isdir(folder):
            # notify the user
            plexus.info.log('target folder {!r} does not exist; creating'.format(folder))
            # were we asked to build all parent directories?
            if self.createPrefix:
                # yes, do it
                os.makedirs(os.path.abspath(folder))
            # otherwise
            else:
                # attempt
                try:
                    # to create the directory
                    os.mkdir(folder)
                # if that fails
                except OSError:
                    # complain
                    return plexus.error.log('could not create folder {!r}'.format(folder))

        # now that it's there, build a local filesystem around it
        pfs = self.vfs.local(root=folder)

        # build a virtual filesystem so we can record the directory layout
        mfs = self.vfs.virtual()
        # here is the directory structure
        mfs['spells'] = mfs.folder()
        mfs['assets'] = mfs.folder()

        # attempt to
        try:
            # realize the layout
            pfs.make(name=merlin.METAFOLDER, tree=mfs)
        # if it fails
        except OSError as error:
            # complain
            return plexus.error.log(str(error))

        # mount it
        self.vfs['/project'] = pfs
        self.vfs['/merlin/project'] = pfs[merlin.METAFOLDER]

        # if a name was not specified
        if self.project is None:
            # use the last portion of the target folder
            _, self.project = os.path.split(os.path.abspath(folder))

        # build a new project description
        project = merlin.newProject(name=self.project)
        # and save it
        merlin.curator.saveProject(project=project)

        # all done
        return 0


# end of file
