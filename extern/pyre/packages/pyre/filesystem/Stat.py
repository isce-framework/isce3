# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import os
import stat


# my base class
from .Recognizer import Recognizer


# class declaration
class Stat(Recognizer):
    """
    This class provides support for sorting out local filesystem entries based on the lowest
    level of metadata available: the actual representation on the hard disk.
    """


    # types
    from .BlockDevice import BlockDevice
    from .CharacterDevice import CharacterDevice
    from .Directory import Directory
    from .File import File
    from .Link import Link
    from .NamedPipe import NamedPipe
    from .Socket import Socket

    # constants
    filetypes = {
        stat.S_IFDIR: Directory,
        stat.S_IFBLK: BlockDevice,
        stat.S_IFCHR: CharacterDevice,
        stat.S_IFREG: File,
        stat.S_IFLNK: Link,
        stat.S_IFIFO: NamedPipe,
        stat.S_IFSOCK: Socket,
        }


    # interface
    @classmethod
    def recognize(cls, entry, follow_symlinks=False):
        """
        The most basic file recognition: convert the name of a file into a {Node} descendant
        and decorate it with all the meta-data available on the disk.
        """
        # attempt to
        try:
            # pull the information from the hard filesystem
            meta = entry.stat(follow_symlinks=follow_symlinks)
        # if something goes wrong
        except (FileNotFoundError, NotADirectoryError) as error:
            # there is nothing further to be done
            return None

        # grab my mode
        mode = meta.st_mode

        # attempt to
        try:
            # lookup the file type and build the meta-data
            info = cls.filetypes[stat.S_IFMT(mode)]
        # if not there
        except KeyError:
            # we have a bug
            import journal
            # build a message
            msg = "'{}': unknown file type: mode={}".format(entry, mode)
            # and complain
            return journal.firewall("pyre.filesystem").log(msg)

        # if successful, build an info node and return it
        return info(uri=entry, info=meta)


# end of file
