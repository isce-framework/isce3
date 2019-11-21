#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that component registration interacts correctly with the pyre configurator model
"""

# access
# print(" -- importing pyre")
import pyre
# print(" -- done")


def declare():

    # declare a protocol
    class protocol(pyre.protocol):
        """a protocol"""
        # properties
        p1 = pyre.properties.str()
        p2 = pyre.properties.str()
        # behavior
        @pyre.provides
        def do(self):
            """behave"""

    # declare a component
    class component(pyre.component, family="test", implements=protocol):
        """a component"""
        # traits
        p1 = pyre.properties.str(default="p1")
        p2 = pyre.properties.str(default="p2")

        @pyre.export
        def do(self):
            """behave"""
            return "component"

    return component


def test():

    # and the model
    model = pyre.executive.nameserver
    # model.dump(pattern='test')

    # print(" -- making some configuration changes")
    # add an assignment
    model['test.p1'] = 'step 1'
    # an alias
    model.alias(alias='p1', target='test.p1')
    # and a reference to the alias
    model['ref'] = '{p1}'
    # check that they point to the same slot
    assert model.retrieve(name='p1') == model.retrieve(name='test.p1')
    # save the nodes
    ref = model.retrieve(name='ref')
    step_0 = model.retrieve(name='test.p1')

    # now declare the component and its protocol
    # print(" -- declaring components")
    component = declare()
    # print(" -- done")

    # model.dump(pattern='')
    assert component.p1 == 'step 1'
    assert component.p2 == 'p2'

    # check that the model is as we expect
    # model.dump()
    assert model['test.p1'] == component.p1
    assert model['test.p2'] == component.p2
    # how about the alias and the reference?
    assert model['ref'] == component.p1
    assert model['p1'] == component.p1

    # make a late registration to what is now the component trait
    model['test.p2'] = 'step 2'
    # model.dump(pattern='test')
    # and check
    assert component.p1 == 'step 1'
    assert component.p2 == 'step 2'

    return



# main
if __name__ == "__main__":
    test()


# end of file
