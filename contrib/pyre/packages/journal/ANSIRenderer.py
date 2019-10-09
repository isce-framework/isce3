# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# packages
import pyre


# the implemented interfaces
from .Renderer import Renderer


# declaration
class ANSIRenderer(pyre.component, family="journal.renderers.ansi", implements=Renderer):
    """
    A color capable text renderer
    """


    # access to the color schemes
    from . import schemes


    # public state
    header = pyre.properties.str(default=">>")
    header.doc = "the marker to use while rendering the diagnostic metadata"

    body = pyre.properties.str(default="--")
    body.doc = "the marker to use while rendering the diagnostic body"

    scheme = schemes.colors(default=schemes.light)
    scheme.doc = "the set of colors to use when rendering diagnostics"


    # interface
    @pyre.provides
    def render(self, page, metadata):
        """
        Convert the diagnostic information into a form that a device can record
        """
        # build the color palette
        palette = self.palette
        # {normal} gets used a lot
        normal = self.colors["normal"]

        # decorate the metadata with color information
        fields = {}
        # the filename
        try:
            filename = metadata["filename"]
        except KeyError:
            fields["filename"] = ''
        else:
            color = palette["filename"]
            fields["filename"] = "{}{}{}:".format(color, filename, normal)
        # the line number
        try:
            line = metadata["line"]
        except KeyError:
            fields["line"] = ''
        else:
            color = palette["line"]
            fields["line"] = "{}{}{}:".format(color, line, normal)

        # the severity of the diagnostic
        severity = metadata["severity"]
        severityColor = palette[severity]
        channel = metadata["channel"]
        channelColor = palette["channel"]
        fields["diagnostic"] = " {}{}({}{}{}){}".format(
            severityColor, severity, channelColor, channel, severityColor, normal)

        # the function
        function = metadata.get("function", '')
        # if we have non-trivial information
        if function:
            # display it
            color = palette["function"]
            fields["function"] = " in {}{!r}{}".format(color, function, normal)
        # otherwise
        else:
            # say nothing
            fields["function"] = ""


        # build the header
        header = " {0.header} {filename}{line}{diagnostic}{function}:".format(self, **fields)
        # return the header
        yield header
        # and the body
        for line in page:
            yield " {} {} ".format(self.body, line)

        # all done
        return


    # meta methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # grab my scheme
        scheme = self.scheme
        # build my palette
        self.palette = {
            trait.name: self.colors[getattr(scheme, trait.name)]
            for trait in self.scheme.pyre_traits()
            }
        # all done
        return


    # private data
    esc = "[{}m"
    colors = {
        "": "", # no color name was given
        "<none>": "", # no color
        "normal": esc.format("0"), # reset back to whatever is the default for the terminal

        # regular colors
        "black": esc.format("0;30"),
        "red": esc.format("0;31"),
        "green": esc.format("0;32"),
        "brown": esc.format("0;33"),
        "blue": esc.format("0;34"),
        "purple": esc.format("0;35"),
        "cyan": esc.format("0;36"),
        "light-gray": esc.format("0;37"),

        # bright colors
        "dark-gray": esc.format("1;30"),
        "light-red": esc.format("1;31"),
        "light-green": esc.format("1;32"),
        "yellow": esc.format("1;33"),
        "light-blue": esc.format("1;34"),
        "light-purple": esc.format("1;35"),
        "light-cyan": esc.format("1;36"),
        "white": esc.format("1;37"),
        }


# end of file
