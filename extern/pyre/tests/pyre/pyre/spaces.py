#!/usr/bin/env python3
# -*- Python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# (c) 1998-2019 all rights reserved
#

# support
import pyre

# build an app
class Spaces(pyre.application):
    """
    Exercise possible public names for traits
    """

    # user configurable state
    flag = pyre.properties.bool(default=True)
    flag.aliases = {"a name with many words"}


    # interface
    def main(self, *args, **kwds):
        """
        The main entry point
        """
        # check that we read the setting from the configuration file correctly
        assert self.flag == False
        # all done
        return 0


# bootstrap
if __name__ == "__main__":
    # instantiate
    app = Spaces(name='spaces')
    # invoke
    status = app.run()
    # share
    raise SystemExit(status)


# end of file
