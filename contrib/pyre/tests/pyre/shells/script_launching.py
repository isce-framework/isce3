#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Instantiate a script
"""


def test():
    # access the framework
    import pyre

    # declare a trivial application
    class application(pyre.application, family='shells.application'):
        """a sample application"""

        @pyre.export
        def main(self): return 0

    # instantiate it
    app = application(name='απόλλων')
    # check that its shell was configured correctly
    assert app.shell.pyre_name == 'λητώ'
    # launch it with script as the default shell
    status = app.run()
    # check the return value
    assert status == 0
    # and return the app
    return app


# main
if __name__ == "__main__":
    test()


# end of file
