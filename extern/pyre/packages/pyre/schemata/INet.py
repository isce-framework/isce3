# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import re # for the parser
import socket


# the base class of internet addresses; useful for detecting address specifications that have
# already been cast to an address instance
class Address:
    """
    Base class for addresses

    This class is useful when attempting to detect whether a value has already been converted
    to an internet address.
    """

    # public data
    @property
    def value(self):
        raise NotImplementedError(
            "class {.__name__!r} must implement 'value'".format(type(self)))


class IPv4(Address):
    """
    Encapsulation of an ipv4 socket address
    """

    # public data
    family = socket.AF_INET
    host = ""
    port = 0

    @property
    def value(self):
        """
        Build the tuple required by {socket.connect}
        """
        return (self.host, self.port)

    # meta methods
    def __init__(self, host='', port=None, **kwds):
        # chain up
        super().__init__(**kwds)
        # store my state
        self.host = host
        self.port = 0 if port is None else int(port)
        # all done
        return

    def __str__(self):
        # easy enough
        return "{!r}:{}".format(self.host, self.port)


class Unix(Address):
    """
    Unix domain sockets
    """

    # public data
    family = socket.AF_UNIX
    path = None

    @property
    def value(self):
        """
        Build the value expected by {socket.connect}
        """
        return self.path

    # meta methods
    def __init__(self, path, **kwds):
        # chain up
        super().__init__(**kwds)
        # save my path
        self.path = path
        # all done
        return

    def __str__(self):
        # easy enough
        return self.path


# the schema type superclass
from .Schema import Schema


# declaration
class INet(Schema):
    """
    A type declarator for internet addresses
    """


    # types
    from .exceptions import CastingError
    # the address base class
    address = Address
    # the more specialized types
    ip = ip4 = ipv4 = IPv4
    unix = local = Unix


    # constants
    any = ip(host='', port=0) # the moral equivalent of zero...
    typename = 'inet' # the name of my type
    complaint = 'could not coerce {0.value!r} into an internet address'


    # interface
    def coerce(self, value, **kwds):
        """
        Attempt to convert {value} into a internet address
        """
        # {address} instances go right through
        if isinstance(value, self.address): return value
        # strings
        if isinstance(value, str):
            # get processes by my parser
            return self.parse(value.strip())
        # anything else is an error
        raise self.CastingError(value=value, description=self.complaint)


    def recognize(self, family, address):
        """
        Return an appropriate address type based on the socket family
        """
        # ipv4
        if family == socket.AF_INET:
            # unpack the raw address
            host, port = address
            # return an ipv4 addres
            return self.ipv4(host=host, port=port)

        # unix
        if family == socket.AF_UNIX:
            # return a unix addres
            return self.unix(path=address)

        # otherwise
        raise NotImplementedError("unsupported socket family: {}".format(family))


    def parse(self, value):
        """
        Convert {value}, expected to be a string, into an inet address
        """
        # interpret an empty {value}
        if not value:
            # as an ip4 address, on the local host at some random port
            return self.ipv4()
        # attempt to match against my regex
        match = self.regex.match(value)
        # if it failed
        if not match:
            # bail out
            raise self.CastingError(value=value, description=self.complaint)

        # check whether this an IP address
        family = match.group('ip')
        # and if so
        if family:
            # invoke the correct constructor
            return getattr(self, family)(host=match.group('host'), port=match.group('port'))

        # check whether this is a UNIX address
        family = match.group('unix')
        # and if so
        if family:
            # invoke the correct constructor
            return getattr(self, family)(path=match.group('path'))

        # if we get this far, there was no explicit family name in the address, in which case
        # just build an ipv4 address
        return self.ipv4(host=match.group('host'), port=match.group('port'))


    def json(self, value):
        """
        Generate a JSON representation of {value}
        """
        # represent as a string
        return self.string(value)


    # meta-methods
    def __init__(self, default=any, **kwds):
        # chain up with my default
        super().__init__(default=default, **kwds)
        # all done
        return


    # private data
    regex = re.compile(
        r"(?P<unix>unix|local):(?P<path>.+)"
        r"|"
        r"(?:(?P<ip>ip|ip4|ip6|ipv4|ipv6):)?(?P<host>[^:]+)(?::(?P<port>[0-9]+))?"
        )


# end of file
