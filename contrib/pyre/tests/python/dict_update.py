#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Check whether dict.update() respects the order imposed by source dict's __iter__ order
"""


class ordered(dict):

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

    source = ordered()
    for n in seq:
        source[n] = None

    destination = ordered()
    destination.update(source)

    target = [key for key in source]

    assert seq == target

    return destination, target


# main
if __name__ == "__main__":
    test()


# end of file
