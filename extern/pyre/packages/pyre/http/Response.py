# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import time # to generate timestamps
import collections # for ordered dict
# my superclass
from ..nexus.exceptions import NexusError


# the base class for all responses
class Response(NexusError):
    """
    The base class for all http server responses.
    """

    # It may appear counter-intuitive to make all responses, including the ones that are
    # correct behavior, derive from {Exception}. The motivation is flow control rather than
    # signaling erroneous conditions. The process of assembling the correct response is rather
    # complex, and the server may need to bail out from arbitrarily deeply in the work flow. I
    # do not think that explicit unwinding of the stack is the right way to go; let python do
    # the work...

    # public data
    code = None # a numeric code indicating the type of HTTP response
    status = '' # a very short description of the type of HTTP response
    server = None # the server that accepted the client request
    headers = None # meta data about the response
    encoding = 'utf-8' # the default encoding for the response payload
    version = (1,1) # the HTTP version spoken here


    # meta-methods
    def __init__(self, server, version=version, encoding=encoding, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the server reference
        self.server = server
        # the version
        self.version = version
        # the encoding
        self.encoding = encoding
        # build the headers
        headers = collections.OrderedDict()
        #  decorate
        headers["Server"] = server.name
        headers["Date"] = self.timestamp()
        # attach
        self.headers = headers
        # all done
        return


    # implementation details
    def timestamp(self, tick=None):
        """
        Generate a conforming timestamp
        """
        # use now if necessary
        if tick is None: tick = time.time()
        # unpack
        year, month, day, hh, mm, ss, wd, y, z = time.gmtime(tick)
        # render and return
        return "{}, {:02} {} {} {:02}:{:02}:{:02} GMT".format(
            self.weekdays[wd], day, self.months[month], year,
            hh, mm, ss
            )


    # private data
    months = (
        None,
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')

    weekdays = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')


# end of file
