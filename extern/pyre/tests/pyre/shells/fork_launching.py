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
    # externals
    import pyre # access the framework

    # declare a trivial application
    class application(pyre.application, family='shells.application'):
        """a sample application"""

        @pyre.export
        def main(self, channels, **kwds):
            """The behavior of the child process"""
            # unpack
            stdout, stderr = channels
            # say something
            stdout.write(b"Hello father!")
            # all done
            return 0


        # implementation details
        @pyre.export
        def launched(self, channels, **kwds):
            """The behavior of the parent process"""
            # unpack the channels to the child
            stdout, stderr = channels
            # make sure we can read its output correctly
            assert stdout.read(13) == b"Hello father!"
            # all done
            return 0


    # instantiate it
    app = application(name='κάστωρ')
    # check that its shell was configured correctly
    assert app.shell.pyre_name == 'λήδα'
    # launch it
    status = app.run()
    # check it
    assert status == 0
    # and return the application instance
    return app


# main
if __name__ == "__main__":
    test()


# end of file
