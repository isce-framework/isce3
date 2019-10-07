#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that we understand the way various locales render numbers
"""


def test_el_GR():
    # print("el_GR")
    import locale
    try:
        locale.setlocale(locale.LC_ALL, "el_GR.UTF-8")
    except locale.Error:
        return

    # test str -> object conversions
    assert 1234567 == locale.atoi("1.234.567")
    assert 1234567.89 == locale.atof("1.234.567,89")

    # test object -> str
    assert "1234,56" == locale.str(1234.56)
    assert "1.234,56" == locale.format_string("%.2f", 1234.56, grouping=True, monetary=True)
    # print(locale.currency(1234567.89, True, True, False))
    # print(locale.currency(1234567.89, True, True, True))
    assert "1.234.567,89 Eu" == locale.currency(1234567.89, True, True, False)
    assert "1.234.567,89 EUR " == locale.currency(1234567.89, True, True, True)

    return


def test_en_US():
    # print("en_US")
    import locale
    try:
        locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
    except locale.Error:
        return

    # test str -> object conversions
    assert 1234567 == locale.atoi("1,234,567")
    assert 1234567.89 == locale.atof("1,234,567.89")

    # test object -> str
    assert "1234.56" == locale.str(1234.56)
    assert "1,234.56" == locale.format_string("%.2f", 1234.56, grouping=True, monetary=True)
    # print(locale.currency(1234567.89, True, True, False))
    # print(locale.currency(1234567.89, True, True, True))
    assert "$1,234,567.89" == locale.currency(1234567.89, True, True, False)
    assert "USD 1,234,567.89" == locale.currency(1234567.89, True, True, True)

    return


def test_en_GB():
    # print("en_GB")
    import locale
    try:
        locale.setlocale(locale.LC_ALL, "en_GB.UTF-8")
    except locale.Error:
        return

    # test str -> object conversions
    assert 1234567 == locale.atoi("1,234,567")
    assert 1234567.89 == locale.atof("1,234,567.89")

    # test object -> str
    assert "1234.56" == locale.str(1234.56)
    assert "1,234.56" == locale.format_string("%.2f", 1234.56, grouping=True, monetary=True)
    # print(locale.currency(1234567.89, True, True, False))
    # print(locale.currency(1234567.89, True, True, True))
    assert "£1,234,567.89" == locale.currency(1234567.89, True, True, False)
    assert "GBP 1,234,567.89" == locale.currency(1234567.89, True, True, True)

    return


def test_fr_FR():
    # print("fr_FR")
    import locale
    try:
        locale.setlocale(locale.LC_ALL, "fr_FR.UTF-8")
    except locale.Error:
        return

    # test str -> object conversions
    assert 1234567 == locale.atoi("1234567")
    assert 1234567.89 == locale.atof("1234567,89")

    # test object -> str
    assert "1234,56" == locale.str(1234.56)
    # print(locale.format("%.2f", 1234.56, grouping=True, monetary=True))
    assert "1234,56" == locale.format_string("%.2f", 1234.56, grouping=False, monetary=True)
    # print(locale.currency(1234567.89, True, True, False))
    # print(locale.currency(1234567.89, True, True, True))
    assert "1 234 567,89 Eu" == locale.currency(1234567.89, True, True, False)
    assert "1 234 567,89 EUR " == locale.currency(1234567.89, True, True, True)

    return


def test():
    # test_el_GR()
    test_en_GB()
    test_en_US()
    test_fr_FR()
    test_el_GR()

    return


# main
if __name__ == "__main__":
    test()


# end of file
