#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

import journal


def info():
    d = journal.info("foo")
    d.active = True
    d.log("here is some information")
    # also from C++
    journal.extension.infoTest("foo")
    return


def warning():
    d = journal.warning("foo")
    d.log("here is a warning")
    # also from C++
    journal.extension.warningTest("foo")
    return


def error():
    d = journal.error("foo")
    d.log("unknown error")
    # also from C++
    journal.extension.errorTest("foo")
    return


def debug():
    d = journal.debug("foo")
    d.active = True
    d.log("unknown error")
    # also from C++
    journal.extension.debugTest("foo")
    return


def firewall():
    d = journal.firewall("foo")
    d.log("unknown error")
    # also from C++
    journal.extension.firewallTest("foo")
    return


# main
if __name__ == "__main__":
    info()
    warning()
    error()
    debug()
    firewall()

# end of file
