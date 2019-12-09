# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# packages
import pyre


# the color scheme protocol
class ColorScheme(pyre.protocol, family="journal.schemes"):
    """
    The color scheme protocol definition
    """

    header = pyre.properties.str()
    body = pyre.properties.str()

    filename = pyre.properties.str()
    line = pyre.properties.str()
    function = pyre.properties.str()
    stackTrace = pyre.properties.str()

    src = pyre.properties.str()

    channel = pyre.properties.str()
    debug = pyre.properties.str()
    firewall = pyre.properties.str()
    info = pyre.properties.str()
    warning = pyre.properties.str()
    error = pyre.properties.str()


# the base component
class NoColor(pyre.component, family="journal.schemes.nocolor", implements=ColorScheme):
    """
    The base color scheme: no coloring
    """

    header = pyre.properties.str()
    body = pyre.properties.str()

    filename = pyre.properties.str()
    line = pyre.properties.str()
    function = pyre.properties.str()
    stackTrace = pyre.properties.str()

    src = pyre.properties.str()

    channel = pyre.properties.str()
    debug = pyre.properties.str()
    firewall = pyre.properties.str()
    info = pyre.properties.str()
    warning = pyre.properties.str()
    error = pyre.properties.str()


# scheme suitable for rendering against dark backgrounds
class DarkBackground(NoColor, family="journal.schemes.dark"):
    """
    A color scheme suitable for rendering in terminals with a dark background
    """

    header = pyre.properties.str(default="light-green")
    body = pyre.properties.str(default="light-green")

    filename = pyre.properties.str(default="light-green")
    line = pyre.properties.str(default="light-green")
    function = pyre.properties.str(default="light-purple")
    stackTrace = pyre.properties.str(default="<none>")

    src = pyre.properties.str(default="yellow")

    channel = pyre.properties.str(default="light-blue")
    debug = pyre.properties.str(default="light-cyan")
    firewall = pyre.properties.str(default="light-red")
    info = pyre.properties.str(default="light-green")
    warning = pyre.properties.str(default="yellow")
    error = pyre.properties.str(default="light-red")


# scheme suitable for rendering against light backgrounds
class LightBackground(NoColor, family="journal.schemes.light"):
    """
    A color scheme suitable for rendering in terminals with a dark background
    """

    header = pyre.properties.str(default="blue")
    body = pyre.properties.str(default="blue")

    filename = pyre.properties.str(default="blue")
    line = pyre.properties.str(default="blue")
    function = pyre.properties.str(default="blue")
    stackTrace = pyre.properties.str(default="<none>")

    src = pyre.properties.str(default="blue")

    channel = pyre.properties.str(default="light-blue")
    debug = pyre.properties.str(default="cyan")
    firewall = pyre.properties.str(default="purple")
    info = pyre.properties.str(default="green")
    warning = pyre.properties.str(default="brown")
    error = pyre.properties.str(default="red")


# aliases
colors = ColorScheme
none = NoColor
dark = DarkBackground
light = LightBackground


# end of file
