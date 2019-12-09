# -*- coding: utf-8 -*-
#
# {project.authors}
# {project.affiliations}
# (c) {project.span} all rights reserved
#


# external
import re
# support
import {project.name}


# declaration
class UX:
    """
    Translate URLs into actions
    """

    # interface
    def dispatch(self, plexus, server, request):
        """
        Analyze the {{request}} received by the {{server}} and invoke the appropriate {{plexus}} request
        """
        # get the request type
        command = request.command
        # get the request uri
        url = request.url
        # take a look
        match = self.regex.match(url)
        # if there is no match
        if not match:
            # something terrible has happened
            return server.responses.NotFound(server=server)

        # find who matched
        token = match.lastgroup
        # look up the handler
        handler = getattr(self, token)
        # invoke
        return handler(plexus=plexus, server=server, request=request, match=match)


    # handlers
    def version(self, plexus, server, **kwds):
        """
        The client requested the version of the server
        """
        # get the version
        version = "{{}}.{{}} build {{}} on {{}}".format(*{project.name}.version(), {project.name}.built())
        # all done
        return server.documents.JSON(server=server, value=version)


    def stop(self, plexus, **kwds):
        """
        The client is asking me to die
        """
        # log it
        plexus.info.log("shutting down")
        # and exit
        raise SystemExit(0)


    def document(self, plexus, server, request, **kwds):
        """
        The client requested a document from the {{plexus}} pfs
        """
        # form the uri
        uri = "/www" + request.url
        # open the document and serve it
        return server.documents.File(uri=uri, server=server, application=plexus)


    def root(self, plexus, server, request, **kwds):
        """
        The client requested the root document
        """
        # form the uri
        uri = "/www/{{0.pyre_namespace}}.html".format(plexus)
        # open the document and serve it
        return server.documents.File(uri=uri, server=server, application=plexus)


    # private data
    regex = re.compile("|".join([
        r"/(?P<version>query/meta/version)",
        r"/(?P<stop>action/meta/stop)",
        r"/(?P<document>(fonts/.+)|(graphics/.+)|(scripts/.+)|(styles/.+)|(.+\.js))",
        r"/(?P<root>.*)",
        ]))


# end of file
