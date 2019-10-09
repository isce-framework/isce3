# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Definitions for all the exceptions raised by this package
"""

from ..framework.exceptions import FrameworkError


# db api 2.0 compliant exception hierarchy
# not my first choice for a classification strategy, but there you go...

class Warning(FrameworkError):
    """
    Exception raised for important warnings, such as data truncation, loss of precision and
    other idications that the implementation engines have carried out a request in a perhaps
    incorrect way
    """


class Error(FrameworkError):
    """
    Base class for all exceptions that are raised by this module to indicate an unrecoverable
    error
    """


class InterfaceError(Error):
    """
    Base class for exceptions raised by the database client code, not the database back end
    """


class DatabaseError(Error):
    """
    Base class for exceptions raised by the database back end
    """


class DataError(DatabaseError):
    """
    Exception raised when a data processing error occurs
    """


class OperationalError(DatabaseError):
    """
    An exception that indicates environmental problems that are generally not related to the
    program itself
    """


class IntegrityError(DatabaseError):
    """
    Exception raised when an operation violates the referential integrity of the data store
    """


class InternalError(DatabaseError):
    """
    Exception raised when the back end reports an internal error
    """


class ProgrammingError(DatabaseError):
    """
    Exception raised when there is a problem with the SQL statement being executed
    """

    # public data
    description = "while executing {0.command!r}: {0.diagnostic}"

    # meta-methods
    def __init__(self, command, diagnostic, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the error info
        self.command = command
        self.diagnostic = diagnostic
        # all done
        return


class NotSupportedError(DatabaseError):
    """
    Exception raised when a method or database API was used that is not supported by the
    database client
    """


# end of file
