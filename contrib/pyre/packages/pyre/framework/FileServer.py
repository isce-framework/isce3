# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import weakref
# pyre types
from .. import primitives, schemata
# superclass
from ..filesystem.Filesystem import Filesystem


# class declaration
class FileServer(Filesystem):
    """
    The manager of the virtual filesystem

    Instances of {FileServer} manage hierarchical namespaces implemented as a virtual
    filesystem. The contents of these namespaces are retrieved using URIs, and can be arbitrary
    objects, although they are typically either local or remote files.

    The framework uses a {FileServer} instance to decouple the logical names of resources from
    their physical locations at runtime. For example, during the bootstrapping process the
    framework looks for user preferences for pyre applications. On Unix like machines, these
    are stored in '~/.pyre' and its subfolders. The entire hierarchy is mounted in the virtual
    filesystem under '/-/user'. This has the following advantages:

    * applications can navigate through the contents of '/-/user' as if it were an actual
      filesystem

    * configuration settings that require references to entries in '/_/user' can now be
      expressed portably, since there is no need to hardwire actual paths

    Applications are encouraged to lay out their own custom namespaces. The application
    developer can refer to resources through their standardized logical names, whereas the user
    is free to provide the mapping that reflects their physical location at runtime.
    """


    # constants
    DOT_PYRE = primitives.path('~/.pyre')
    USER_DIR = primitives.path('/__pyre/user')
    STARTUP_DIR = primitives.path('/__pyre/startup')
    PACKAGES_DIR = primitives.path('/__pyre/packages')


    # public data
    @property
    def systemFolders(self):
        """
        Return the sequence of uris of the {pyre} system folders
        """
        # first the startup folder
        yield self.STARTUP_DIR
        # next the user folder
        yield self.USER_DIR
        # finally the packages folder
        yield self.PACKAGES_DIR
        # all done
        return


    # interface
    def open(self, uri, **kwds):
        """
        Convert {uri} into an input stream
        """
        # make sure {uri} is a {schemata.uri} instance
        uri = schemata.uri().coerce(uri)
        # get the {uri} scheme
        scheme = uri.scheme

        # if {scheme} is missing, assume it is a file from the local filesystem
        if scheme is None or scheme == 'file':
            # so attempt to
            try:
                # open it and return the associated file object
                return open(uri.address, **kwds)
            # if {uri} is not in my logical namespace
            except self.NotFoundError as error:
                # complain
                raise self.SourceNotFoundError(filesystem=self, node=error.node, uri=uri)
            # if {uri} maps to a non-existent file
            except IOError:
                # complain
                raise self.SourceNotFoundError(filesystem=self, node=None, uri=uri)
            # if {uri} maps to a folder
            except OSError: # NYI: after python3.3: convert to IsADirectoryError
                # complain
                raise self.IsFolderError(filesystem=self, node=None, uri=uri)

        # if the scheme is {vfs}
        if scheme == 'vfs':
            # assuming the uri is within my virtual filesystem
            try:
                # get the node
                node = self[uri.address]
            # if {uri} is not in my logical namespace
            except self.NotFoundError as error:
                # complain
                raise self.SourceNotFoundError(filesystem=self, node=error.node, uri=uri)

            # if the node is a folder
            if node.isFolder:
                # complain
                raise self.IsFolderError(filesystem=self, node=node, uri=uri)
            # otherwise, open it
            return node.open(**kwds)

        # if i didn't recognize the {scheme}, complain
        raise self.URISpecificationError(uri=uri, reason="unsupported scheme {!r}".format(scheme))


    # convenience: access to the filesystem factories
    def local(self, root, **kwds):
        """
        Build a local filesystem
        """
        # access the factory
        from .. import filesystem
        # invoke it
        return filesystem.local(root=str(root), **kwds)


    def virtual(self, **kwds):
        """
        Build a virtual filesystem
        """
        # access the factory
        from .. import filesystem
        # invoke it
        return filesystem.virtual(**kwds)


    # framework support
    def initializeNamespace(self):
        """
        Construct the initial layout of my virtual filesystem
        """
        # at boot time, we target two directories: the user's private configuration folder, and
        # the current working directory. mounting either one may fail: the folder may not
        # exist, or it may not have the correct permissions. so we have to be careful. the
        # runtime relies on the existence of the virtual folders, so me mount empty folders on
        # failure

        # get the current working directory; it is guaranteed to be an absolute path so there
        # is no reason to resolve it

        startup = primitives.path.cwd()
        # however, it is possible that it doesn't exist, or it has the wrong permissions; try to
        # mount it, or mount an empty folder if anything goes wrong
        self[self.STARTUP_DIR] = self.retrieveFilesystem(root=startup)

        # the user's private folder, typically at {~/.pyre}
        userdir = self.DOT_PYRE.expanduser().resolve()
        # same deal: mount it or an empty folder
        self[self.USER_DIR] = self.retrieveFilesystem(root=userdir)

        # build the virtual directory where packages park their configuration
        self[self.PACKAGES_DIR] = self.folder()

        # all done
        return


    def registerPackage(self, package):
        """
        Make the package configuration folder accessible in the virtual filesystem
        """
        # This should be done very carefully because multiple packages may share a common
        # installation folder. For example, this is true of the packages that ship with the
        # standard pyre distribution. The registration procedure takes care not to mount
        # redundant local filesystems in the virtual namespace to make sure that {vfs} nodes
        # that are related to each other are resolved by the same local filesystem

        # sign in
        # print("pyre.framework.FileServer:")
        # get the package name
        name = package.name
        # show me
        # print("  package name: {}".format(name))

        # attempt to
        try:
            # print("  looking for {}".format(self.USER_DIR / name))
            # hunt down the package directory in the user area
            userdir = self[self.USER_DIR / name]
        # if not there
        except self.NotFoundError:
            # nothing to do: the user directory was discovered at level 1 during boot, so the
            # directory really does not exist
            # print("    not there")
            pass
        # if it is there
        else:
            # print("    found it; exploring...")
            # look deeply
            userdir.discover()

        # grab the package prefix
        prefix = package.prefix
        # print("  package prefix: {}".format(prefix))
        # not much to do if there isn't one
        if not prefix: return package

        # otherwise, mount/get the associated filesystem
        fs = self.retrieveFilesystem(root=prefix)
        # attempt to
        try:
            # print("  looking for {}".format(package.DEFAULTS))
            # look for the configuration folder
            defaults = fs[package.DEFAULTS]
        # if not there
        except fs.NotFoundError:
            # print("    not there")
            # nothing else to do
            return package
        # print("    found it")

        # if the configuration folder is empty
        if not defaults.contents:
            # it might be because we haven't explored it yet
            # print("    expanding it")
            defaults.discover(levels=1)

        # look for configuration files
        for encoding in self.executive.configurator.encodings():
            # build the configuration file name
            filename = "{}.{}".format(name, encoding)
            # look for
            try:
                # the node that corresponds to the configuration file
                cfgfile = defaults[filename]
            # if it's not there
            except fs.NotFoundError:
                # bail
                continue
            # if it is there, mount it within the package directory
            # print("  mounting {}".format(self.PACKAGES_DIR / filename))
            self[self.PACKAGES_DIR / filename] = cfgfile

        # look for the configuration folder
        try:
            # get the associated node
            # print("  looking for {}/{}".format(package.DEFAULTS, name))
            cfgdir = defaults[name]
        # if it's not there
        except fs.NotFoundError:
            # print("    not there")
            # no problem
            pass
        # if it is there
        else:
            # attach it
            # print("    mounting it")
            self[self.PACKAGES_DIR / name] = cfgdir.discover()

        # all done
        return package


    # meta-methods
    def __init__(self, executive=None, **kwds):
        # chain up
        super().__init__(**kwds)
        # remember my executive
        self.executive = None if executive is None else weakref.proxy(executive)
        # initialize the table of known mount points
        self.mounts = {}
        # all done
        return


    # implementation details
    def retrieveFilesystem(self, root, levels=1):
        """
        Retrieve {root} if it is an already mounted filesystem; if not, mount it and return it
        """
        # check whether
        try:
            # i have seen this path before
            fs = self.mounts[root]
        # if i haven't
        except KeyError:
            # no problem
            pass
        # if i have
        else:
            # i promised to look, so do it; we assume that {discover} is safe to call on a
            # previously explored filesystem
            return fs.discover(levels=levels)

        # next, we go through the mount points of all filesystems that have been seen
        # previously and check whether the path supplied by the caller is a child of one of my
        # mounted filesystems; if it is, mount the missing directories to reach the target
        # mount point; if not, mount it as a new filesystem
        for path in self.mounts:
            # attempt to
            try:
                # figure out whether it is contained in the tree rooted at {path}
                diff = root.relativeTo(path)
            # if not
            except ValueError:
                # no problem, grab the next one
                continue

            # we got one; get the associated folder node where we will park {root}
            folder = self.mounts[path]
            # go through each level in the relative path
            for level in diff.parts:
                # try to
                try:
                    # get the associated folder
                    folder = folder[level]
                # if this fails
                except self.NotFoundError:
                    # and the the folder has any contents
                    if folder.contents:
                        # {level} does not exist; not much else we can do but punt
                        return self.folder()
                    # otherwise, it must be that the folder hasn't been explored yet
                    folder.discover(levels=1)
                    # look again
                    try:
                        # for the subdirectory
                        folder = folder[level]
                    # if it fails again after discovery
                    except self.NotFoundError:
                        # {level} does not exist; not much else we can do but punt
                        return self.folder()
            # if we get this far, we have what we are looking for
            return folder.discover(levels=levels)

        # it wasn't there; let's try
        try:
            # to make it
            folder = self.local(root=root).discover(levels=levels)
        # if that fails
        except self.GenericError:
            # make an empty folder
            folder = self.folder()

        # if all goes well; remember it
        self.mounts[root] = folder
        # and return it
        return folder


# end of file
