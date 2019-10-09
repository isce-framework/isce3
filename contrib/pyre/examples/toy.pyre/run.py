#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# get the toys
import toy

# driver
def test():

    # startup stuff
    # print("person.friends:\n  {}\n  {}".format(toy.person.friends, type(toy.person.friends)))
    # print("student.friends:\n  {}\n  {}".format(toy.student.friends, type(toy.student.friends)))

    # make a person
    michael = toy.person(name='michael')
    # he is the default teacher of all students
    assert michael is toy.student.friends['teacher']

    # here is a generic student
    jane = toy.student(name='jane')
    jane.activities = toy.relax(), 'study#math', '#physics'

    activities = tuple(jane.perform())
    assert activities == (
        'relaxing for 1.0 hour', 'studying for 1.0 hour', 'studying for 2.0 hours'
        )
    assert jane.friends['teacher'] is michael # are class defaults inherited?

    # create persons named in the configuration file
    alec = toy.student(name='alec')
    activities = tuple(alec.perform())
    assert activities == (
        'studying for 2.0 hours', 'relaxing for 3.0 hours',
        'studying for 1.0 hour', 'relaxing for 1.5 hours'
        )
    # print(alec.friends)
    # print(alec.friends['girlfriend'].pyre_name)
    # print(alec.friends['girlfriend'].friends)
    assert alec.friends['teacher'] is michael

    ally = toy.student(name='ally')
    activities = tuple(ally.perform())
    assert activities == (
        'studying for 0.5 hours', 'relaxing for 3.0 hours',
        'studying for 1.5 hours', 'relaxing for 1.5 hours'
        )
    # print(ally.friends)
    assert ally.friends['teacher'] is michael

    # check the relationships
    # print(alec.friends['girlfriend'])
    assert alec.friends['girlfriend'] is ally
    # print(ally.friends['boyfriend'])
    assert ally.friends['boyfriend'] is alec

    joe = toy.policeman(name='joe')
    activities = tuple(joe.perform())
    assert activities == ('patrolling for 5.0 hours', 'relaxing for 1.5 hours')

    # augment some relationships
    alec.friends.update({'joe': "#joe"})
    alec.friends['joe'] is joe

    # show me
    # print("person.friends:\n  {}\n  {}".format(toy.person.friends, type(toy.person.friends)))
    # print("student.friends:\n  {}\n  {}".format(toy.student.friends, type(toy.student.friends)))
    # print("alec.friends:\n  {}\n  {}".format(alec.friends, type(alec.friends)))
    # print("ally.friends:\n  {}\n  {}".format(ally.friends, type(ally.friends)))

    # all done
    return jane, alec, ally, joe


# main
if __name__ == "__main__":
    test()


# end of file
