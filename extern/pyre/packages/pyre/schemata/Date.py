# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import datetime
# superclass
from .Schema import Schema


# my declaration
class Date(Schema):
    """
    A type declarator for dates
    """


    # constants
    format = "%Y-%m-%d" # the default date format
    typename = 'date' # the name of my type


    # interface
    def coerce(self, value, **kwds):
        """
        Attempt to convert {value} into a date
        """
        # treat false values as uninitialized
        if not value: return None

        # check whether {value} is already a {date} instance
        if isinstance(value, datetime.date):
            # in which case we are done
            return value
        # perhaps it is a {datetime} instance
        if isinstance(value, datetime.datetime):
            # in which case extract its date component
            return value.date()

        # the rest assumes that {value} is a string; attempt
        try:
            # to strip the value
            value = value.strip()
        # if this fails
        except AttributeError:
            # complain
            raise self.CastingError(value=value, description=self.complaint)
        # if there is nothing left
        if not value:
            # bail
            return None

        # attempt to
        try:
            # cast it into a date
            return datetime.datetime.strptime(value, self.format).date()
        # if this fails
        except (AttributeError, TypeError, ValueError) as error:
            # complain
            raise self.CastingError(value=value, description=str(error))


    def string(self, value):
        """
        Render value as a string that can be persisted for later coercion
        """
        # respect {None}
        if value is None: return None
        # my value knows
        return value.strftime(self.format)


    def json(self, value):
        """
        Generate a JSON representation of {value}
        """
        # represent as a string
        return self.string(value)


    # meta-methods
    def __init__(self, default=datetime.date.today(), format=format, **kwds):
        # chain up with my default
        super().__init__(default=default, **kwds)
        # store the format
        self.format = format
        # all done
        return


# end of file
