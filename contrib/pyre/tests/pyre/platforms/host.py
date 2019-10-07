#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise host configuration
"""


def test():
    # support
    import pyre, journal

    # derive an application class
    class app(pyre.component):
        """sample application"""

        # my host
        host = pyre.platforms.platform()


    # instantiate
    one = app(name='one')
    # get the platform manager
    host = one.host
    # check i have one
    assert host

    # make a channel
    channel = journal.debug('pyre.host')
    # if the channel is active
    if channel:
        # show me some details
        channel.line('host: {}'.format(host))
        channel.line('         hostname: {.hostname}'.format(host))
        channel.line('         nickname: {.nickname}'.format(host))
        channel.line('     architecture: {.cpus.architecture}'.format(host))
        channel.line('          sockets: {.cpus.sockets}'.format(host))
        channel.line('            cores: {.cpus.cores}'.format(host))
        channel.line('             cpus: {.cpus.cpus}'.format(host))
        channel.line('         platform: {.platform}'.format(host))
        channel.line('          release: {.release}'.format(host))
        channel.line('         codename: {.codename}'.format(host))
        channel.line('     distribution: {.distribution}'.format(host))
        channel.line('  package manager: {.packager.name}'.format(host))
        channel.log()

    # and return
    return one


# main
if __name__ == "__main__":
    test()


# end of file
