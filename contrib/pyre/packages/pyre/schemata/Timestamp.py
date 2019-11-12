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
class Timestamp(Schema):
    """
    A type declarator for timestamps
    """


    # constants
    format = "%Y-%m-%d %H:%M:%S" # the default format
    typename = 'timestamp' # the name of my type
    complaint = 'could not coerce {0.value!r} into a time'


    # interface
    def coerce(self, value, **kwds):
        """
        Attempt to convert {value} into a timestamp
        """
        # perhaps {value} is already a {datetime} instance
        if isinstance(value, datetime.datetime):
            # in which case just return it
            return value

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
            # assume it is a string; strip it and covert it
            return datetime.datetime.strptime(value, self.format)
        # if anything goes wrong
        except Exception as error:
            # complain
            raise self.CastingError(value=value, description=self.complaint)


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
    def __init__(self, default=datetime.datetime.today(), format=format, **kwds):
        # chain up with my default
        super().__init__(default=default, **kwds)
        # store the format
        self.format = format
        # all done
        return


# end of file
