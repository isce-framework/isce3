#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


import pyre


# a protocol
class activity(pyre.protocol, family="sample.activities"):
    """the activity specification"""

    # get the time units
    from pyre.units import time

    # my traits
    duration = pyre.properties.dimensional(default=time.hour)

    # my public interface
    @pyre.provides
    def do(self):
        """do something"""

    # set  up my default implementation
    @classmethod
    def pyre_default(cls, **kwds): return relax


# a few implementations
class study(pyre.component, family="sample.activities.study", implements=activity):
    """an activity"""

    # my traits
    duration = pyre.properties.dimensional(default=4*activity.time.hour)

    @pyre.export
    def do(self): return "studying"

class relax(pyre.component, family="sample.activities.relax", implements=activity):
    """an activity"""

    # my traits
    duration = pyre.properties.dimensional(default=2*activity.time.hour)

    @pyre.export
    def do(self): return "relaxing"

class sleep(pyre.component, family="sample.activities.sleep", implements=activity):
    """an activity"""

    # my traits
    duration = pyre.properties.dimensional(default=8*activity.time.hour)

    @pyre.export
    def do(self): return "sleeping"


# the container
class person(pyre.component, family="sample.person"):
    """a component container"""
    activities = pyre.properties.list(schema=activity())



def test():
    # easy access to time units
    from pyre.units.time import hour

    # make a container; configuration comes from {sample.pml}
    alec = person('alec')
    # dump
    # print('alec:')
    # for task in alec.activities:
        # print('  {}'.format(task))
        # print('    {} for {:base={scale},label=hours}'.format(
                # task.do(), task.duration, scale=activity.time.hour))

    # here is what we expect
    # task 0
    task = alec.activities[0]
    assert task.pyre_name == 'physics'
    assert task.pyre_family() == 'sample.activities.study'
    assert task.duration == .5*hour

    # task 1
    task = alec.activities[1]
    assert task.pyre_name == 'wow'
    assert task.pyre_family() == 'sample.activities.relax'
    assert task.duration == 1*hour

    # task 2
    task = alec.activities[2]
    assert task.pyre_name == 'nap'
    assert task.pyre_family() == 'sample.activities.sleep'
    assert task.duration == 3*hour

    return


# main
if __name__ == "__main__":
    test()


# end of file
