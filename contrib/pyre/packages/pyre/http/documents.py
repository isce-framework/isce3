# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import json
# the base class
from .Response import Response


# base class for all normal responses
class OK(Response):
    """
    OK
    """
    # state
    code = 200
    status = __doc__
    description = "Request fulfilled, document follows"

    # interface
    def render(self, **kwds):
        """
        Generate the payload
        """
        # nothing to do
        return b''


# simple documents
class Literal(OK):
    """
    A response built out of a literal string
    """

    # public data
    encoding = 'utf-8' # the encoding to use when converting to bytes

    # interface
    def render(self, server, **kwds):
        """
        Pack the contents of the file into a binary buffer
        """
        # return my value as a byte stream
        return self.value.encode(self.encoding)

    # meta-methods
    def __init__(self, value, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the value
        self.value = value
        # all done
        return


class JSON(OK):
    """
    A response built out of the JSON encoding of a python object
    """

    # public data
    encoding = 'utf-8' # the encoding to use when converting to bytes

    # interface
    def render(self, **kwds):
        """
        Encode the object in JSON format
        """
        # return my value as a byte stream
        return json.dumps(self.value).encode(self.encoding)

    # meta-methods
    def __init__(self, value, **kwds):
        # chain up
        super().__init__(**kwds)
        # add the content type to the headers
        self.headers['Content-Type'] = f'application/json; charset={self.encoding}'
        # save the value
        self.value = value
        # all done
        return


# document responses
class Document(OK):
    """
    A response built out of an application generated document
    """

    # meta-methods
    def __init__(self, application, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the application context so we can render the document
        self.application = application
        # all done
        return


class File(Document):
    """
    A document response built out of a file in the application private document root
    """

    # public data
    uri = None # the file to serve

    # interface
    def render(self, server, **kwds):
        """
        Pack the contents of the file into a binary buffer
        """
        # get the uri
        uri = self.uri
        # and the application
        app = self.application
        # attempt to
        try:
            # open the file
            stream = app.pfs[uri].open(mode='rb')
        # if something goes wrong
        except app.pfs.GenericError:
            # raise something bad
            raise server.responses.NotFound(server=server)
        # all is well, send it along
        return stream.read()

    # meta-methods
    def __init__(self, uri, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the uri
        self.uri = uri
        # all done
        return


# end of file
