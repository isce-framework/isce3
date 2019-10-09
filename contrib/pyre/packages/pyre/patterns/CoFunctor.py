# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# class declaration
class CoFunctor:
    """
    Converts a functor into a coroutine
    """


    # generator interface
    def send(self, value):
        """
        Accept a value
        """
        # pass it to my generator
        return self.generator.send(value)


    def throw(self, errorTp, error=None, traceback=None):
        """
        Raise an exception
        """
        # pass it along
        return self.generator.throw(errorTp, error, traceback)


    def close(self):
        """
        Shut things down
        """
        # pass it along
        return self.generator.close()


    # meta-methods
    def __init__(self, *args, **kwds):
        # chain up
        super().__init__(*args, **kwds)
        # build my generator
        self.generator = self.__call__()
        # and prime it
        next(self.generator)
        # all done
        return


    def __next__(self):
        """
        Advance
        """
        return next(self.generator)


# end of file
