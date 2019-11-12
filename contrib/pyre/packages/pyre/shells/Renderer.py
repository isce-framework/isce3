# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import pyre
# access to my protocol
import journal.protocols
# my facilities
from .Terminal import Terminal


# declaration
class Renderer(pyre.component, family='pyre.shells.renderer',
               implements=journal.protocols.renderer):
    """
    Custom replacement for the {journal} renderer
    """


    # public state
    terminal = Terminal()


    # interface
    @pyre.export
    def render(self, page, metadata):
        """
        Convert the diagnostic information into a form that a device can record
        """
        # if the page is empty, there's nothing to do
        if not page: return

        # my colors; hardwired for now
        marker = self.palette[metadata['severity']]
        blue = self.terminal.ansi['blue']
        normal = self.terminal.ansi['normal']

        # extract the information from the metadata
        # channel = '{}{}{}'.format(blue, metadata['channel'], normal)
        # severity = '{}{}{}'.format(marker, self.severityShort(metadata), normal)
        # decorate the first line
        # yield "{}: {}: {}".format(channel, severity, page[0])

        # an alternate formatting
        channel = '{}{}{}'.format(marker, metadata['channel'], normal)
        # decorate the first line
        yield "{}: {}".format(channel, page[0])

        # and render the rest
        yield from page[1:]

        # all done
        return


    def __init__(self, **kwds):
        super().__init__(**kwds)

        # get my terminal
        terminal = self.terminal
        # build my palette
        self.palette = {
            'info': terminal.rgb256(red=0, green=2, blue=0),
            'warning': terminal.rgb256(red=5, green=3, blue=0),
            'error': terminal.rgb256(red=5, green=0, blue=0),
            'debug': terminal.rgb256(red=1, green=3, blue=5),
            'firewall': terminal.ansi['light-red'],
            }

        # all done
        return


    # implementation details
    def severityUpcase(self, metadata):
        """
        Return an uppercased severity
        """
        # easy...
        return metadata['severity'].upper()


    def severityShort(self, metadata):
        """
        Return a three letter abbreviation for the severity
        """
        # easy
        return self.shortSeverities[metadata['severity']]


    # private data
    shortSeverities = {
        'info': 'INF',
        'warning': 'WRN',
        'error': 'ERR',
        'debug': 'DBG',
        'firewall': 'FIREWALL', # don't shorten this one
        }


# end of file
