#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Sanity check: verify that the exceptions defined in the module are accessible
"""


def test():
    # access the relevant modules
    from pyre.extensions import postgres as pyrepg
    import pyre.db.exceptions as exceptions
    # initialize the module exceptions manually
    # this is normally done automatically when a client first requests a connection to the
    # database back end
    pyrepg.registerExceptions(exceptions)

    # now check
    # make sure the exception object is accessible
    warning = pyrepg.Warning
    # make sure it is decorated correctly
    assert warning.__name__ == 'Warning'
    assert warning.__module__ == 'pyre.db.exceptions'
    assert warning.__bases__ == (exceptions.FrameworkError,)
    # verify it can be caught
    try:
        raise pyrepg.Warning('a generic database warning')
    except pyrepg.Warning as warning:
        pass

    # make sure the exception object is accessible
    error = pyrepg.Error
    # make sure it is decorated correctly
    assert error.__name__ == 'Error'
    assert error.__module__ == 'pyre.db.exceptions'
    assert error.__bases__ == (exceptions.FrameworkError,)
    # verify it can be caught
    try:
        raise pyrepg.Error('a generic database error')
    except pyrepg.Error as error:
        pass

    # make sure the exception object is accessible
    error = pyrepg.InterfaceError
    # make sure it is decorated correctly
    assert error.__name__ == 'InterfaceError'
    assert error.__module__ == 'pyre.db.exceptions'
    assert error.__bases__ == (pyrepg.Error,)
    # verify it can be caught
    try:
        raise pyrepg.InterfaceError('a generic database error')
    except pyrepg.InterfaceError as error:
        pass

    # make sure the exception object is accessible
    error = pyrepg.DatabaseError
    # make sure it is decorated correctly
    assert error.__name__ == 'DatabaseError'
    assert error.__module__ == 'pyre.db.exceptions'
    assert error.__bases__ == (pyrepg.Error,)
    # verify it can be caught
    try:
        raise pyrepg.DatabaseError('a generic database error')
    except pyrepg.DatabaseError as error:
        pass

    # make sure the exception object is accessible
    error = pyrepg.DataError
    # make sure it is decorated correctly
    assert error.__name__ == 'DataError'
    assert error.__module__ == 'pyre.db.exceptions'
    assert error.__bases__ == (pyrepg.DatabaseError,)
    # verify it can be caught
    try:
        raise pyrepg.DataError('a generic database error')
    except pyrepg.DataError as error:
        pass

    # make sure the exception object is accessible
    error = pyrepg.OperationalError
    # make sure it is decorated correctly
    assert error.__name__ == 'OperationalError'
    assert error.__module__ == 'pyre.db.exceptions'
    assert error.__bases__ == (pyrepg.DatabaseError,)
    # verify it can be caught
    try:
        raise pyrepg.OperationalError('a generic database error')
    except pyrepg.OperationalError as error:
        pass

    # make sure the exception object is accessible
    error = pyrepg.IntegrityError
    # make sure it is decorated correctly
    assert error.__name__ == 'IntegrityError'
    assert error.__module__ == 'pyre.db.exceptions'
    assert error.__bases__ == (pyrepg.DatabaseError,)
    # verify it can be caught
    try:
        raise pyrepg.IntegrityError('a generic database error')
    except pyrepg.IntegrityError as error:
        pass

    # make sure the exception object is accessible
    error = pyrepg.InternalError
    # make sure it is decorated correctly
    assert error.__name__ == 'InternalError'
    assert error.__module__ == 'pyre.db.exceptions'
    assert error.__bases__ == (pyrepg.DatabaseError,)
    # verify it can be caught
    try:
        raise pyrepg.InternalError('a generic database error')
    except pyrepg.InternalError as error:
        pass

    # make sure the exception object is accessible
    error = pyrepg.ProgrammingError
    # make sure it is decorated correctly
    assert error.__name__ == 'ProgrammingError'
    assert error.__module__ == 'pyre.db.exceptions'
    assert error.__bases__ == (pyrepg.DatabaseError,)
    # verify it can be caught
    try:
        raise pyrepg.ProgrammingError(diagnostic='a generic database error', command='none')
    except pyrepg.ProgrammingError as error:
        pass

    # make sure the exception object is accessible
    error = pyrepg.NotSupportedError
    # make sure it is decorated correctly
    assert error.__name__ == 'NotSupportedError'
    assert error.__module__ == 'pyre.db.exceptions'
    assert error.__bases__ == (pyrepg.DatabaseError,)
    # verify it can be caught
    try:
        raise pyrepg.NotSupportedError('a generic database error')
    except pyrepg.NotSupportedError as error:
        pass

    return


# main
if __name__ == "__main__":
    test()


# end of file
