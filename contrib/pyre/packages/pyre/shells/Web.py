# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import pyre
# my superclass
from .Script import Script


# declaration
class Web(Script, family='pyre.shells.web'):
    """
    A shell enables application interactivity over the web
    """

    # user configurable state
    auto = pyre.properties.bool(default=True)
    auto.doc = 'controls whether to automatically launch the browser'

    # a marker that enables applications to deduce the type of shell that is hosting them
    model = pyre.properties.str(default='web')
    model.doc = "the programming model"


    # interface
    @pyre.export
    def launch(self, application, *args, **kwds):
        """
        Invoke the {application} behavior
        """
        # before doing anything else, let's check whether the user has asked for help on the
        # command line; if so, just invoke the application help behavior and exit
        # get the nameserver
        nameserver = self.pyre_nameserver
        # go through the markers
        for marker in self.helpon:
            # if it is known by the configuration store
            if marker in nameserver:
                # get help
                return application.help(*args, **kwds)

        # ok, we are in business; create a nexus
        nexus = pyre.nexus.node(name="{.pyre_name}.nexus".format(application))
        # attach it to the application
        application.nexus = nexus
        # register it with the nexus
        nexus.services['web'] = 'http'
        # activate
        nexus.prepare(application=application)
        # get the web server
        web = nexus.services['web']
        # get the address of the web server
        address = web.address
        # show me
        application.info.log('web server on {}'.format(address))
        # if we were asked to launch a browser
        if self.auto:
            # grab the package with the browser selection logic
            import webbrowser
            # form a url
            url = 'http://localhost:{.port}/'.format(address)
            # launch the browser
            webbrowser.open(url)

        # set up a net
        try:
            # get the nexus to do its thing
            # N.B. this is an infinite loop; it is the responsibility of the application to
            # terminate the interaction with the user and exit gracefully
            status = nexus.watch()
        # if the user interrupted
        except KeyboardInterrupt as event:
            # launch the handler
            return application.pyre_interrupted(info=event)

        # if all went well
        application.pyre_shutdown(status=status)

        # in any case, we are all done
        return status


# end of file
