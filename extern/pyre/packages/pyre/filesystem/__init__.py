# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# external
import re
# support
from .. import primitives


# factories
# filesystems
def virtual(**kwds):
    """
    Build a virtual filesystem
    """
    from .Filesystem import Filesystem
    return Filesystem(**kwds)


def local(root, listdir=None, recognizer=None, **kwds):
    """
    Build a local filesystem, i.e. one that encapsulates the contents of a filesystem that is
    mounted on the machine that hosts this process.

    parameters:
        {root}: the directory to use as the root of the new filesystem
        {listdir}: the mechanism that lists the contents of directories
        {recognizer}: the mechanism that identifies the types of files
    """
    # build a walker if necessary
    listdir = walker() if listdir is None else listdir
    # build a recognizer
    recognizer = stat() if recognizer is None else recognizer


    # ensure that {root} is an absolute path so that we can protect the filesystem
    # representation in case the application manipulates the current working directory of the
    # process
    root = primitives.path(root).resolve()
    # grab the location metadata
    info = recognizer.recognize(root)

    # if the location doesn't exist
    if not info:
        # complain
        raise MountPointError(uri=root, error='mount point not found')

    # if the root is a directory
    if info.isFolder:
        # access the local filesystem factory
        from .Local import Local
        # build one
        return Local(metadata=info, walker=listdir, recognizer=recognizer, **kwds)

    # perhaps it is a zipfile
    import zipfile
    # so check, and if so
    if zipfile.is_zipfile(str(root)):
        # access the zip filesystem factory
        from .Zip import Zip
        # build one and return it
        return Zip(metadata=info)

    # out of ideas
    raise MountPointError(uri=root, error='invalid mount point')


def zip(root, **kwds):
    """
    Attempt to build a zip filesystem out of {root}, which is expected to be a zip archive
    """
    # ensure {root} is an absolute path, just in case the application changes the current
    # working directory
    root = primitives.path(root).resolve()
    # check whether the location exists
    if not root.exists():
        # and if not, complain
        raise MountPointError(uri=root, error='mount point not found')

    # access the zip package
    import zipfile
    # if {root} does not point to a valid archive
    if not zipfile.is_zipfile(str(root)):
        # complain
        raise MountPointError(uri=root, error="mount point is not a zipfile")

    # otherwise, get a recognizer to build the metadata for the archive
    info = stat().recognize(root)
    # access the zip filesystem factory
    from .Zip import Zip
    # build one and return it
    return Zip(metadata=info)


# nodes
def naked(**kwds):
    """
    Build a naked node, i.e. a node that is a trivial wrapper around a file in the local
    filesystem. Useful for quick and dirty mounting of specific files without having to build a
    local filesystem out of the folder that contains them
    """
    # get the constructor
    from .Naked import Naked
    # build one and return it
    return Naked(**kwds)


# explorers
def finder(**kwds):
    """
    Build an explorer that traverses the contents of filesystems and returns ({node}, {name})
    pairs that match an optional {pattern}
    """
    from .Finder import Finder
    return Finder(**kwds)

def treeExplorer(**kwds):
    """
    Build an explorer that creates a report of filesystem contents formatted like a tree
    """
    from .TreeExplorer import TreeExplorer
    return TreeExplorer(**kwds)

def simpleExplorer(**kwds):
    """
    Build an explorer that creates a simple report of filesystem contents
    """
    from .SimpleExplorer import SimpleExplorer
    return SimpleExplorer(**kwds)


# tools for poking around locally mounted filesystems
def stat():
    """
    Build a recognizer of the contents of local filesystems based on {os.stat}
    """
    from .Stat import Stat
    return Stat

def walker():
    """
    Build an object that can list the contents of locally mounted folders
    """
    from .Walker import Walker
    return Walker


# exceptions thrown by the factories
from .exceptions import MountPointError


# debugging support:
#     import the package and set to something else, e.g. pyre.patterns.ExtentAware
#     to change the runtime behavior of these objects
_metaclass_Node = type

def debug():
    """
    Support for debugging the filesystem package
    """
    # print(" ++ debugging 'pyre.filesystem'")
    # Attach ExtentAware as the metaclass of Node and Filesystem so we can verify that all
    # instances of these classes are properly garbage collected
    from ..patterns.ExtentAware import ExtentAware
    global _metaclass_Node
    _metaclass_Node = ExtentAware
    # all done
    return


# end of file
