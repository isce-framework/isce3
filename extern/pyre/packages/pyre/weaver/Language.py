# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import pyre


# declaration
class Language(pyre.protocol, family="pyre.weaver.languages"):
    """
    The protocol specification for output languages
    """


    # constants
    # the language normalization table
    languages = {
        "c++": "cxx",
        "fortran": "f77",
        "fortran77": "f77",
        }


    # framework hooks
    @classmethod
    def pyre_convert(cls, value, **kwds):
        # if {value} is a string
        if isinstance(value, str):
            # convert to lower case
            language = value.lower()
            # and translate
            return cls.languages.get(language, language)
        # otherwise, I have nothing to say
        return value


    # interface
    @pyre.provides
    def render(self):
        """
        Render the document
        """


    @pyre.provides
    def header(self):
        """
        Render the header of the document
        """


    @pyre.provides
    def body(self):
        """
        Render the body of the document
        """


    @pyre.provides
    def footer(self):
        """
        Render the footer of the document
        """


# end of file
