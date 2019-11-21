#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that checking compatibility among protocols produces the correct reports
"""


def test():
    import pyre

    # declare a couple of protocols
    class base(pyre.protocol):
        """the base protocol"""
        common = pyre.property()

    class derived(base):
        """a derived one, so automatically compatible"""
        extra = pyre.property()

    class ok(pyre.protocol):
        """one that doesn't derive but has the right public protocol"""
        common = pyre.property()

    class notok(pyre.protocol):
        """one that doesn't provide the right public protocol"""
        what = pyre.property()

    class badtype(pyre.protocol):
        """one that has the right trait but of the wrong type"""
        @pyre.provides
        def common(self):
            """method, not property"""

    class shadow(base):
        """one that derives but shadows the trait in an incompatible way"""
        @pyre.provides
        def common(self):
            """method, not property"""

    # collect the traits
    base_common = base.pyre_trait("common")
    derived_common = derived.pyre_trait("common")
    derived_extra = derived.pyre_trait("extra")
    ok_common = ok.pyre_trait("common")
    notok_what = notok.pyre_trait("what")
    badtype_common = badtype.pyre_trait("common")
    shadow_common = shadow.pyre_trait("common")

    # compatibility checks
    # these ones should succeed
    assert derived.pyre_isCompatible(base)
    assert ok.pyre_isCompatible(base)
    assert derived.pyre_isCompatible(ok)

    # now the ones that should fail
    report = ok.pyre_isCompatible(derived, fast=False)
    assert not report
    assert len(report.incompatibilities) == 1
    error = report.incompatibilities[derived_extra][0]
    assert isinstance(error, ok.TraitNotFoundError)

    report = notok.pyre_isCompatible(base, fast=False)
    assert not report
    assert len(report.incompatibilities) == 1
    error = report.incompatibilities[base_common][0]
    assert isinstance(error, notok.TraitNotFoundError)

    report = notok.pyre_isCompatible(derived, fast=False)
    assert not report
    assert len(report.incompatibilities) == 2
    error = report.incompatibilities[derived_common][0]
    assert isinstance(error, notok.TraitNotFoundError)
    error = report.incompatibilities[derived_extra][0]
    assert isinstance(error, notok.TraitNotFoundError)

    report = notok.pyre_isCompatible(ok, fast=False)
    assert not report
    assert len(report.incompatibilities) == 1
    error = report.incompatibilities[ok_common][0]
    assert isinstance(error, notok.TraitNotFoundError)

    report = badtype.pyre_isCompatible(base, fast=False)
    assert not report
    assert len(report.incompatibilities) == 1
    error = report.incompatibilities[base_common][0]
    assert isinstance(error, badtype.CategoryMismatchError)

    report = badtype.pyre_isCompatible(derived, fast=False)
    assert not report
    assert len(report.incompatibilities) == 2
    error = report.incompatibilities[derived_common][0]
    assert isinstance(error, badtype.CategoryMismatchError)
    error = report.incompatibilities[derived_extra][0]
    assert isinstance(error, badtype.TraitNotFoundError)

    report = badtype.pyre_isCompatible(ok, fast=False)
    assert not report
    assert len(report.incompatibilities) == 1
    error = report.incompatibilities[ok_common][0]
    assert isinstance(error, badtype.CategoryMismatchError)

    report = shadow.pyre_isCompatible(base, fast=False)
    assert not report
    assert len(report.incompatibilities) == 1
    error = report.incompatibilities[base_common][0]
    assert isinstance(error, shadow.CategoryMismatchError)

    report = shadow.pyre_isCompatible(derived, fast=False)
    assert not report
    assert len(report.incompatibilities) == 2
    error = report.incompatibilities[derived_common][0]
    assert isinstance(error, shadow.CategoryMismatchError)
    error = report.incompatibilities[derived_extra][0]
    assert isinstance(error, shadow.TraitNotFoundError)

    report = shadow.pyre_isCompatible(ok, fast=False)
    assert not report
    assert len(report.incompatibilities) == 1
    error = report.incompatibilities[ok_common][0]
    assert isinstance(error, shadow.CategoryMismatchError)

    return base, derived, ok, notok, badtype, shadow


# main
if __name__ == "__main__":
    test()


# end of file
