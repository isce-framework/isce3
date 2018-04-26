#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from libcpp cimport bool
from libcpp.string cimport string

from Poly2d cimport Poly2d
from Metadata cimport Metadata

cdef extern from "isce/core/ResampSlc.h" namespace "isce::core":

    # ResampSlc class
    cdef cppclass ResampSlc:

        # Constructor
        ResampSlc() except +

        # Polynomial getters
        Poly2d rgCarrier()
        Poly2d azCarrier()
        Poly2d doppler()
        # Polynomial setters
        void rgCarrier(Poly2d &)
        void azCarrier(Poly2d &)
        void doppler(Poly2d &)

        # Get metadata
        Metadata metadata()
        Metadata refMetadata()
        # Set metadata
        void metadata(Metadata)
        void refMetadata(Metadata)

        # Get/set number of lines per processing tile
        size_t linesPerTile()
        void linesPerTile(size_t)

        # Main resamp entry point
        void resamp(const string &, const string &, const string &,
                    const string &, bool, bool, int)

# end of file
