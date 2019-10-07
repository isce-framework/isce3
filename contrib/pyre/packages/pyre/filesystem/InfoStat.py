# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import stat
import time


# declaration
class InfoStat:
    """
    Mixin that knows how to pull information from {stat} structures
    """

    # constants
    # entities
    user = object()
    group = object()
    other = object()
    # access
    read = object()
    write = object()
    execute = object()


    # interface
    # decode the permissions
    def mode(self, entity=user, access=read):
        """
        Determine whether {entity} has read permissions
        """
        # deduce the mask
        mask = self.masks[(entity, access)]
        # and extract the permissions
        return True if mask & self.permissions else False


    # meta methods
    def __init__(self, info, **kwds):
        # chain up
        super().__init__(**kwds)
        # if we were not handed any information
        if not info:
            # complain for now
            raise NotImplementedError(f"'{self.uri}': stat node with no info")

        # file attributes
        self.uid = info.st_uid
        self.gid = info.st_gid
        self.size = info.st_size
        self.permissions = stat.S_IMODE(info.st_mode)
        # timestamps
        self.accessTime = info.st_atime
        self.creationTime = info.st_ctime
        self.modificationTime = info.st_mtime
        # all done
        return


    # implementation details
    masks = {
        (user, read): stat.S_IRUSR,
        (user, write): stat.S_IWUSR,
        (user, execute): stat.S_IXUSR,
        (group, read): stat.S_IRGRP,
        (group, write): stat.S_IWGRP,
        (group, execute): stat.S_IXGRP,
        (other, read): stat.S_IROTH,
        (other, write): stat.S_IWOTH,
        (other, execute): stat.S_IXOTH,
    }


    # debugging support
    def dump(self, channel, indent=''):
        """
        Place the known information about this file in the given {channel}
        """
        channel.line(f"{indent}node {self}")
        channel.line(f"{indent}  uri: '{self.uri}'")
        channel.line(f"{indent}  uid: {self.uid}")
        channel.line(f"{indent}  gid: {self.gid}")
        channel.line(f"{indent}  size: {self.size}")
        channel.line(f"{indent}  permissions: {self.permissions:o}")
        channel.line(f"{indent}  access time: {time.ctime(self.accessTime)}")
        channel.line(f"{indent}  creation time: {time.ctime(self.creationTime)}")
        channel.line(f"{indent}  modification time: {time.ctime(self.modificationTime)}")
        # all done
        return


# end of file
