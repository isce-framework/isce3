# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# class declaration
class Dependency:
    """
    Mix-in class that enables a node to notify its observers when its value changes
    """


    # value management
    def setValue(self, value):
        """
        Override the value setter to notify my observers that my value changed
        """
        # pass the value along
        super().setValue(value)
        # notify my observers
        self.flush()
        # all done
        return self


# end of file
