# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access the framework
import pyre


# tasks
class task(pyre.protocol, family='toy.tasks'):

    # types
    from pyre.units.time import hour

    # user configurable state
    duration = pyre.properties.dimensional(default=1*hour)

    # interface
    @pyre.provides
    def perform(self):
        """do something"""

    # framework support
    @classmethod
    def pyre_default(cls, **kwds):
        """the default task"""
        return relax


# the base class for task implementers
class activity(pyre.component, implements=task):

    duration = pyre.properties.dimensional(default=1*task.hour)

    @pyre.export
    def perform(self):
        return '{0.description} for {0.duration:base={hour},label=hour|hours}'.format(
            self, hour=task.hour)

# a few tasks
class relax(activity, family='toy.tasks.relax'):

    description = 'relaxing'


class study(activity, family='toy.tasks.study'):

    description = 'studying'
    duration = pyre.properties.dimensional(default=2*task.hour)


class patrol(activity, family='toy.tasks.patrol'):

    description = 'patrolling'
    duration = pyre.properties.dimensional(default=1.5*task.hour)


# people
class people(pyre.protocol, family='toy.people'):

    activities = pyre.properties.list(schema=task())

    @pyre.provides
    def perform(self):
        """perform my specified activities"""

    @classmethod
    def pyre_default(cls, **kwds):
        """the default implementation"""
        return person


# persons are people; or is it the other way around?
class person(pyre.component, family='toy.people.person', implements=people):

    friends = pyre.properties.dict(schema=people())
    activities = pyre.properties.list(schema=task())

    @pyre.export
    def perform(self):
        """perform my activities"""
        # print("{.pyre_spec!r}:".format(self))
        # print("  friends:")
        # for name, friend in self.friends.items():
            # print("    {!r}: {.pyre_spec!r}".format(name, friend))
            #print("    {0.pyre_spec!r}: {0}".format(friend))
        # print("  activities:", self.activities)
        # for activity in self.activities:
            # print("    {0.pyre_spec!r}: {0}".format(activity))
        # perform each activity
        for activity in self.activities: yield activity.perform()
        # all done
        return


# students
class student(person, family='toy.people.student'):

    activities = pyre.properties.list(schema=task(default=study))


# policemen
class policeman(person, family='toy.people.policeman'):

    activities = pyre.properties.list(schema=task(default=patrol))


# end of file
