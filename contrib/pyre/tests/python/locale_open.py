#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that we can extract a correct default encoding for opening files from the users
environment. Necessary because python gets this wrong by default
"""


def test():
    import locale

    # find out what the default locale is
    print("default locale:", locale.getdefaultlocale())
    # find out what the default preferred encoding is
    print("preferred encoding:", locale.getpreferredencoding())

    locale.setlocale(locale.LC_ALL, '')
    language, encoding = locale.getlocale()
    print("locale: language={0}, encoding={1}".format(language, encoding))

    count = 10
    for line in open("locale_open.py", encoding=encoding):
        print(line, end='')
        count -= 1
        if not count:
            break

    return


# main
if __name__ == "__main__":
    test()


# end of file
