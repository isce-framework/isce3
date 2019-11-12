# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import os
# access the pyre framework
import pyre


# class declaration; merlin is a plexus app
class Merlin(pyre.plexus, family='merlin.components.plexus'):
    """
    The merlin executive and application wrapper
    """


    # constants
    pyre_namespace = 'merlin'
    # types
    from .Action import Action as pyre_action
    # exceptions
    from .exceptions import MerlinError, SpellNotFoundError


    # constants
    METAFOLDER = '.merlin'
    PATH = ['vfs:/merlin/project', 'vfs:/merlin/user', 'vfs:/merlin/system']


    @property
    def searchpath(self):
        """
        Build a list of unique package names from my ancestry in mro order
        """
        # MGA: this should be unnecessary after spells gather themselves under the the correct
        # folder in the {pfs}
        path = set()
        # the project library
        yield 'merlin/project'
        # the user's library
        yield 'merlin/user'
        # and the builtins
        yield 'merlin/system'

        # plus whatever else pyre applications are supposed to do
        yield from super().searchpath

        # all done
        return


    # plexus obligations
    @pyre.export
    def help(self, **kwds):
        """
        Hook for the application help system
        """
        # get the package
        import merlin
        # grab a channel
        channel = self.info
        # set the indentation
        indent = ' '*4
        # make some space
        channel.line()
        # get the help header and display it
        channel.line(merlin.meta.header)

        # reset the pile of actions
        actions = []
        # get the documented commands
        for uri, name, action, tip in self.pyre_action.pyre_documentedActions():
            # and put them on the pile
            actions.append((name, tip))
        # if there were any
        if actions:
            # figure out how much space we need
            width = max(len(name) for name, _ in actions)
            # introduce this section
            channel.line('commands:')
            # for each documented action
            for name, tip in actions:
                # show the details
                channel.line('{}{:>{}}: {}'.format(indent, name, width, tip))
            # some space
            channel.line()

        # flush
        channel.log()
        # and indicate success
        return 0


    # schema factories
    def newProject(self, name):
        """
        Create a new project description object
        """
        # access the class
        from ..schema.Project import Project
        # build the object
        project = Project(name=name)
        # and return it
        return project


    # meta methods
    def __init__(self, name, **kwds):
        # chain up
        super().__init__(name=name, **kwds)

        # the spell manager is built during the construction of superclass; local alias
        self.spellbook = self.pyre_repertoir

        # the curator
        from .Curator import Curator
        self.curator = Curator(name=name+".curator")

        # the asset classifiers
        from .PythonClassifier import PythonClassifier
        self.assetClassifiers = [
            PythonClassifier(name=name+'.python')
            ]

        # all done
        return


    # framework requests
    def pyre_mountApplicationFolders(self, pfs, prefix):
        """
        Build my private filesystem
        """
        # chain up
        pfs = super().pyre_mountApplicationFolders(pfs=pfs, prefix=prefix)

        # check whether the project folder is already mounted
        try:
            # by looking for it within my private file space
            pfs['project']
        # if it's not there
        except pfs.NotFoundError:
            # no worries; we'll go hunting
            pass
        # otherwise, it is already mounted; bug?
        else:
            # DEBUG: remove this when happy it never gets called
            raise NotImplementedError('NYI: multiple attempts to initialize the merlin vfs')

        # check whether we are within a project
        root, metadir = self.locateProjectRoot()

        # get the file server
        vfs  = self.vfs
        # build the project folder
        project = vfs.local(root=root).discover() if root else vfs.folder()
        # build the folder with the merlin metadata
        metadata = vfs.local(root=metadir).discover() if metadir else vfs.folder()

        # mount them
        vfs['project'] = project
        pfs['project'] = metadata

        # and return
        return pfs


    # support
    def pyre_newRepertoir(self):
        """
        Build my spell manager
        """
        # access the factory
        from .Spellbook import Spellbook
        # make one and return it
        return Spellbook(protocol=self.pyre_action)


    def locateProjectRoot(self, folder=None):
        """
        Check whether {folder} is contained within a {merlin} project
        """
        # default to checking starting with the current directory
        folder = os.path.abspath(folder) if folder else os.getcwd()
        # loop until we reach the root of the filesystem
        while folder != os.path.sep:
            # form the path to the {.merlin} subdirectory
            metadir = os.path.join(folder, self.METAFOLDER)
            # if it exists
            if os.path.isdir(metadir):
                # got it
                return folder, metadir
            # otherwise, split the path and try again
            folder, _ = os.path.split(folder)
        # if the loop exited normally, we ran up to the root without success; return
        # empty-handed
        return None, None


# end of file
