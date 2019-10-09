#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Check whether the construct
    for x in dict:
respects the order imposed by the dict's __iter__
"""


class ordered(dict):

    def sort(self):
        self.order = sorted(self.order)

    def __init__(self):
        super().__init__()
        self.order = []
        return

    def __setitem__(self, key, value):
        if key in self:
            return
        self.order.append(key)
        return super().__setitem__(key, value)

    def __iter__(self):
        return iter(self.order)


def test():

    import random
    N = 1000

    seq = random.sample(range(1000000), N)

    data = ordered()
    for i in seq:
        data[i] = None
    data.sort()

    target = [key for key in data]

    assert sorted(seq) == target

    return data


# main
if __name__ == "__main__":
    test()


# end of file
