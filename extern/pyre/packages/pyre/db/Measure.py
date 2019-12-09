# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import time
from . import literals
# superclass
from .. import records
from .. import schemata


# declaration
@schemata.typed
class Measure(records.measure):
    """
    The base class for table field descriptors
    """


    # types
    from .ForeignKey import ForeignKey as foreign


    # field decorations
    def setDefault(self, value):
        """
        Set a new default value
        """
        # install the new default
        self.default = value
        # enable chaining
        return self


    def primary(self):
        """
        Mark a field as a primary key
        """
        # mark
        self._primary = True
        # primary keys do not get default values
        self.default = None
        # and return
        return self


    def unique(self):
        """
        Mark a field as containing values that are unique across the table rows
        """
        # mark
        self._unique = True
        # and return
        return self


    def notNull(self):
        """
        Mark a field as not accepting a NULL value
        """
        # mark
        self._notNull = True
        # and return
        return self


    def references(self, **kwds):
        """
        Mark a field as a foreign key
        """
        # use the specification to create a field reference object and record it
        self._foreign = self.foreign(**kwds)
        # and return
        return self


    # the base class of the local mixins
    class measure:
        """
        The base class of the local mixins.

        Its purpose is to trap value coercion and skip it for the special values {NULL} and
        {DEFAULT} that show up as {literals} instances
        """

        # interface
        def coerce(self, value, **kwds):
            # the literals
            if value is literals.null or value is literals.default:
                # require no processing
                return value
            # for the rest, chain up...
            return super().coerce(value=value, **kwds)



    # mixins for the various supported types
    class bool(measure):
        """Mixin for booleans"""

        # public data
        decl = 'BOOLEAN' # SQL rendering of my type name

        # interface
        def sql(self, value):
            """SQL rendering of {value}"""
            # easy enough
            return 'true' if value else 'false'


    class date(measure):
        """Mixin for dates"""

        # public data
        decl = 'DATE' # SQL rendering of my type name

        # interface
        def sql(self, value):
            """SQL rendering of {value}"""
            # if {value} is a time struct
            if isinstance(value, time.struct_time):
                # use my format to convert it a string
                value = time.strftime(self.format, value)
            # other types of values just get passed along, for now; firewall this later
            # make sure the result is quoted in an SQL compliant way
            return "'{}'".format(value)

        # meta-methods
        def __init__(self, default=None, **kwds):
            # chain up
            super().__init__(default=default, **kwds)
            # all done
            return


    class decimal(measure):
        """Mixin for fixed point numbers"""

        # public data
        @property
        def decl(self):
            """SQL rendering of my type name"""
            # easy enough
            return "DECIMAL({}, {})".format(self.precision, self.scale)

        # interface
        def sql(self, value):
            """SQL rendering of {value}"""
            # convert the decimal into a string
            return str(value)

        # meta-methods
        def __init__(self, precision, scale, **kwds):
            # chain up
            super().__init__(**kwds)
            # save my parts
            self.scale = scale
            self.precision = precision
            # all done
            return


    class float(measure):
        """Mixin for floating point numbers"""

        # public data
        decl = 'DOUBLE PRECISION' # SQL rendering of my type name

        # interface
        def sql(self, value):
            """SQL rendering of my value"""
            # easy enough
            return str(value)


    class int(measure):
        """Mixin for integers"""

        # public data
        decl = 'INTEGER' # SQL rendering of my type name

        # interface
        def sql(self, value):
            """SQL rendering of my value"""
            # easy enough
            return str(value)


    class str(measure):
        """Mixin for strings"""

        # public data
        @property
        def decl(self):
            """SQL rendering of my type name"""
            # get my size
            size = self.maxlen
            # build a size dependent representation
            return 'TEXT' if size is None else 'VARCHAR({})'.format(size)

        # interface
        def sql(self, value):
            """SQL rendering of my value"""
            # easy enough, but must escape any embedded single quotes
            return "'{}'".format(value.replace("'", "''"))

        # meta-methods
        def __init__(self, maxlen=None, **kwds):
            # chain up
            super().__init__(**kwds)
            # save my size
            self.maxlen = maxlen
            # all done
            return


    class time(measure):
        """Mixin for timestamps"""

        # public data
        @property
        def decl(self):
            """SQL rendering of my type name"""
            # with or without timezone
            return 'TIMESTAMP WITH{} TIME ZONE'.format('' if self.timezone else 'OUT')

        # interface
        def sql(self, value):
            """SQL rendering of {value}"""
            # if {value} is a time struct
            if isinstance(value, time.struct_time):
                # use my format to convert it a string
                value = time.strftime(self.format, value)
            # other types of values just get passed along, for now; firewall this later
            # make sure the result is quoted in an SQL compliant way
            return "'{}'".format(value)

        # meta-methods
        def __init__(self, default=None, timezone=False, **kwds):
            # chain up
            super().__init__(default=default, **kwds)
            # timezone support
            self.timezone = timezone
            # all done
            return


    # sql rendering
    def decldefault(self):
        """
        Invoked by the SQL mill to create the declaration of the default value
        """
        # get my default value
        value = self.default
        # if one was not specified
        if value is None:
            # return an empty string
            return ''
        # build an SQL representation for the default value
        value = 'NULL' if value is literals.null else self.sql(value)
        # render
        return " DEFAULT {}".format(value)


    # private data
    # the following markers interpret {None} as 'unspecified'
    _primary = None # am i a primary key?
    _unique = None # are my values unique across the rows of the table?
    _notNull = None # do i accept NULL as a value?
    _foreign = None # foreign key: a tuple (foreign_table, field_descriptor)


# end of file
