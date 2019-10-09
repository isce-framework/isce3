# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# class declaration
class Walker:
    """
    Class that encapsulates listing the contents of a local directory
    """


    # exceptions
    from .exceptions import DirectoryListingError


    # interface
    @classmethod
    def walk(cls, path):
        """
        Assume {path} is a directory, get the names of its contents and iterate over them
        """
        # attempt
        try:
            # to get the contents
             return path.contents
        # if this fails
        except OSError as error:
            # raise a package specific exception
            raise cls.DirectoryListingError(uri=path, error=str(serror))


# end of file
