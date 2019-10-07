# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import itertools
# support
from .. import primitives
# base class
from .Node import Node


# declaration
class Folder(Node):
    """
    The base class for filesystem entries that are containers of other entries

    {Node} and {Folder} are the leaf and container types for the composite that enables the
    representation of the hierarchical structure of filesystems.
    """


    # constants
    isFolder = True


    # types
    # my metadata
    from .InfoFolder import InfoFolder as metadata
    # exceptions
    from .exceptions import (
        FolderInsertionError, NotRootError, FolderError, IsFolderError, NotFoundError)


    # interface
    def open(self):
        """
        Return an iterable over my contents
        """
        # return my contents
        return self.contents.items()


    def mkdir(self, name, parents=True, exist_ok=True):
        # delegate to my filesystem
        return self.filesystem().mkdir(parent=self, name=name, parents=parents, exist_ok=exist_ok)


    def remove(self, node, name=None, **kwds):
        """
        Remove {node} from my contents and its filesystem
        """
        # if the node is a folder
        if node.isFolder:
            # we don't support that yet
            raise NotImplemntedError("NYI: removing directories is not implemented yet")

        # if we were not told the name by which the node is known
        if name is None:
            # look for it in my contents
            for name, child in self.contents.items():
                # look for a match
                if child is node:
                    # bail, we got the name
                    break
            # if we failed to find it
            else:
                # ignore it, for now?
                return
        # otherwise
        else:
            # verify that the node and the name match
            assert self.contents[name] is node

        # ask filesystem to update its persistent store
        self.filesystem().unlink(node=node, **kwds)
        # remove it from my contents
        del self.contents[name]

        # all done
        return


    def write(self, name, contents, mode='w'):
        """
        Create a file with the given {name} and {contents}
        """
        # delegate to my filesystem
        return self.filesystem().write(parent=self, name=name, contents=contents, mode=mode)


    # searching for specific contents
    def find(self, pattern):
        """
        Generate pairs ({node}, {name}) that match the given pattern

        By default, {find} will create a generator that visits the entire contents of the tree
        rooted at this folder. In order to restrict the set of matching names, provide a
        regular expression as the optional argument {pattern}
        """
        # access the finder factory
        from . import finder
        # to build one
        f = finder()
        # and start the search
        return f.explore(folder=self, pattern=pattern)


    # populating filesystems
    def discover(self, **kwds):
        """
        Fill my contents by querying whatever external resource my filesystem represents
        """
        # punt to the implementation in my filesystem
        return self.filesystem().discover(root=self, **kwds)


    # making entire filesystems available through me
    def mount(self, uri, filesystem):
        """
        Make the root of {filesystem} available as {uri} within my filesystem
        """
        # easy enough: just insert {filesystem} at {uri}
        return self._insert(uri=primitives.path(uri), node=filesystem)


    # node factories
    def node(self):
        """
        Build a new node within my filesystem
        """
        # easy enough
        return Node(filesystem=self.filesystem())


    def folder(self):
        """
        Build a new folder within my filesystem
        """
        # also easy
        return Folder(filesystem=self.filesystem())


    # a factory for paths
    @staticmethod
    def path(*args):
        """
        Assemble {args} into a path
        """
        # easy enough
        return primitives.path(*args)


    # meta methods
    def __init__(self, **kwds):
        """
        Build a folder. See {pyre.filesystem.Node} for construction parameters
        """
        # chain up
        super().__init__(**kwds)
        # initialize my contents
        self.contents = {}
        # and return
        return


    def __iter__(self):
        """
        Return an iterator over my {contents}
        """
        # easy enough
        return iter(self.contents)


    def __getitem__(self, uri):
        """
        Retrieve a node given its {uri} as the subscript
        """
        # invoke the implementation and return the result
        return self._retrieve(uri=primitives.path(uri))


    def __setitem__(self, uri, node):
        """
        Attach {node} at {uri}
        """
        # invoke the implementation and return the result
        return self._insert(node=node, uri=primitives.path(uri))


    def __contains__(self, uri):
        """
        Check whether {uri} is one of my children
        """
        # convert
        uri = primitives.path(uri)
        # starting with me
        node = self
        # attempt to
        try:
            # iterate over the names
            for name in uri.names: node = node.contents[name]
        # if node is not a folder, report failure
        except AttributeError: return False
        # if {name} is not among the contents of node, report failure
        except KeyError: return False
        # if we get this far, report success
        return True


    # implementation details
    def _retrieve(self, uri):
        """
        Locate the entry with address {uri}
        """
        # starting with me
        node = self
        # attempt to
        try:
            # hunt down the target node
            for name in uri.names: node = node.contents[name]
        # if any of the folder lookups fail
        except KeyError:
            # notify the caller
            raise self.NotFoundError(
                filesystem=self.filesystem(), node=self, uri=uri, fragment=name)
        # if one of the intermediate nodes is not a folder
        except AttributeError:
            # notify the caller
            raise self.FolderError(
                filesystem=self.filesystem(), node=self, uri=uri, fragment=node.uri)
        # otherwise, return the target node
        return node


    def _insert(self, node, uri, metadata=None):
        """
        Attach {node} at the address {uri}, creating all necessary intermediate folders.
        """
        # starting with me
        current = self
        # make an iterator over the directories in the parent of {uri}
        names = uri.parent.names
        # visit all the levels in {uri}
        for name in names:
            # attempt to
            try:
                # get the node associated with this name
                current = current.contents[name]
            # if not there
            except KeyError:
                # we have reached the edge of the contents of the filesystem; from here on,
                # every access to the contents of {current} will raise an exception and send us
                # here; so all we have to do is build folders until we exhaust {name} and
                # {names}
                for name in itertools.chain((name,), names):
                    # make a folder
                    folder = current.folder()
                    # attach it
                    current.contents[name] = folder
                    # inform the filesystem
                    current.filesystem().attach(node=folder, uri=(current.uri / name))
                    # and advance the cursor
                    current = folder
                # we should have exhausted the loop iterator so there should be no reason
                # to break out of the outer loop; check anyway
                assert tuple(names) == ()
            # if the {current} node doesn't have {contents}
            except AttributeError:
                # complain
                raise self.FolderError(
                    filesystem=current.filesystem(), node=current, uri=uri, fragment=name)

        # at this point, {current} points to the folder that should contain our {node}; get its
        # name by asking the input {uri}
        name = uri.name
        # attempt to
        try:
            # insert it into the contents of the folder
            current.contents[name] = node
        # if the {current} node doesn't have {contents}
        except AttributeError:
            # complain
            raise self.FolderInsertionError(
                filesystem=current.filesystem(), node=node, target=current.uri.name, uri=uri)
        # inform the filesystem
        current.filesystem().attach(node=node, uri=(current.uri / name), metadata=metadata)
        # and return the {node}
        return node


# end of file
