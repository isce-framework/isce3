# -*- coding: utf-8 -*-
#
# {project.authors}
# {project.affiliations}
# (c) {project.span} all rights reserved
#


# access the pyre framework
import pyre
# and my package
import {project.name}
# my action protocol
from .Action import Action


# declaration
class Plexus(pyre.plexus, family='{project.name}.components.plexus'):
    """
    The main action dispatcher
    """

    # types
    from .Action import Action as pyre_action


    # pyre framework hooks
    # support for the help system
    def pyre_banner(self):
        """
        Generate the help banner
        """
        # show the license header
        return {project.name}.meta.license


    # interactive session management
    def pyre_interactiveSessionContext(self, context=None):
        """
        Go interactive
        """
        # prime the execution context
        context = context or {{}}
        # grant access to my package
        context['{project.name}'] = {project.name}  # my package
        # and chain up
        return super().pyre_interactiveSessionContext(context=context)


    # virtual filesystem configuration
    def pyre_mountApplicationFolders(self, pfs, prefix):
        """
        Explore the installation folders and construct my private filesystem
        """
        # chain up
        pfs = super().pyre_mountApplicationFolders(pfs=pfs, prefix=prefix)
        # get my namespace
        namespace = self.pyre_namespace

        # gingerly, look for the web document root; the goal here is to avoid expanding parts
        # of the local filesystem that we don't care about and not descend unnecessarily into
        # potentially arbitrarily deep directory structures; starting at the topmost level
        docroot = prefix
        # descend into the following subdirectories in turn
        for name in ['web', 'www', namespace]:
            # grab the contents
            docroot.discover(levels=1)
            # attempt to
            try:
                # look for the next directory
                docroot = docroot[name]
            # if it's not there
            except prefix.NotFoundError:
                # complain
                self.warning.log("missing web directory; disabling the web app")
                # make sure there is no dispatcher so we know there is no web app support
                self.urlDispatcher = None
                # and bail
                break
        # if all goes well
        else:
            # expand the directory structure below the document root and mount it
            pfs['www'] = docroot.discover()
            # get the dispatcher
            from .UX import UX
            # instantiate and attach
            self.urlDispatcher = UX()

        # all done
        return pfs


    # shells
    def pyre_respond(self, server, request):
        """
        Fulfill a request from an HTTP {{server}}
        """
        # get my dispatcher
        dispatcher = self.urlDispatcher
        # if we don't have one
        if dispatcher is None:
            # everything is an error
            return server.responses.NotFound(server=server)
        # otherwise, refresh my understanding of my document root
        self.pfs['www'].discover()
        # and get it to do its thing
        return dispatcher.dispatch(plexus=self, server=server, request=request)


    # private data
    nexus = None # the event dispatcher
    urlDispatcher = None # converter of urls to actions


# end of file
