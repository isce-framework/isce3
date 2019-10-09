# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import pickle
# the framework
import merlin


# declaration
class Curator(merlin.component, family="merlin.components.curator"):
    """
    The component that manages the project persistent store
    """


    # constants
    projectURI = merlin.properties.str(default='/merlin/project/project.pickle')
    projectURI.doc = 'the location of the persistent project state'


    # interface
    def loadProject(self):
        """
        Retrieve the project configuration information from the archive
        """
        # the fileserver
        vfs = self.vfs
        # get the project pickle
        project = vfs[self.projectURI]
        # retrieve the project instance from the file
        return self.load(node=project)


    def saveProject(self, project):
        """
        Save the given project configuration to the archive
        """
        # pickle the project information into the associated file
        self.save(tag="project", item=project)
        # and return
        return self


    def saveAsset(self, asset):
        """
        Save the given asset to the archive
        """
        # compute the asset tag
        tag = self.vfs.join('assets', asset.name)
        # pickle the project information into the associated file
        self.save(tag=tag, item=asset)
        # and return
        return self


    # implementation details
    def load(self, node):
        """
        Retrieve an object from the merlin file identified by {tag}
        """
        # open the associated file; the caller is responsible for catching any exceptions
        store = node.open(mode="rb")
        # retrieve the object from the store
        item = pickle.load(store)
        # and return it
        return item


    def save(self, tag, item):
        """
        Pickle {item} into the merlin file indicated by {tag}
        """
        # verify that the project directory exists and is mounted; the caller is responsible
        # for catching any exceptions
        folder = self.vfs["/merlin/project"]
        # build the filename associated with {tag}
        vname = "{}.pickle".format(tag)
        # look for the file
        try:
            # careful: this overwrites existing files
            store = folder[vname].open(mode="wb")
        # if not there, create it
        except folder.NotFoundError:
            # FIXME - FILESERVER: this steps outside the file server abstraction, since file
            # creation is not supported yet
            # build the path to the file
            path = folder.uri / vname
            # and open it in write-binary mode
            store = path.open(mode="wb")
        # pickle the item
        pickle.dump(item, store)
        # and return
        return


# end of file
