#[[
CMake - Cross Platform Makefile Generator
Copyright 2000-2019 Kitware, Inc. and Contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of Kitware, Inc. nor the names of Contributors
  may be used to endorse or promote products derived from this
  software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

------------------------------------------------------------------------------

The following individuals and institutions are among the Contributors:

* Aaron C. Meadows <cmake@shadowguarddev.com>
* Adriaan de Groot <groot@kde.org>
* Aleksey Avdeev <solo@altlinux.ru>
* Alexander Neundorf <neundorf@kde.org>
* Alexander Smorkalov <alexander.smorkalov@itseez.com>
* Alexey Sokolov <sokolov@google.com>
* Alex Merry <alex.merry@kde.org>
* Alex Turbov <i.zaufi@gmail.com>
* Andreas Pakulat <apaku@gmx.de>
* Andreas Schneider <asn@cryptomilk.org>
* André Rigland Brodtkorb <Andre.Brodtkorb@ifi.uio.no>
* Axel Huebl, Helmholtz-Zentrum Dresden - Rossendorf
* Benjamin Eikel
* Bjoern Ricks <bjoern.ricks@gmail.com>
* Brad Hards <bradh@kde.org>
* Christopher Harvey
* Christoph Grüninger <foss@grueninger.de>
* Clement Creusot <creusot@cs.york.ac.uk>
* Daniel Blezek <blezek@gmail.com>
* Daniel Pfeifer <daniel@pfeifer-mail.de>
* Enrico Scholz <enrico.scholz@informatik.tu-chemnitz.de>
* Eran Ifrah <eran.ifrah@gmail.com>
* Esben Mose Hansen, Ange Optimization ApS
* Geoffrey Viola <geoffrey.viola@asirobots.com>
* Google Inc
* Gregor Jasny
* Helio Chissini de Castro <helio@kde.org>
* Ilya Lavrenov <ilya.lavrenov@itseez.com>
* Insight Software Consortium <insightsoftwareconsortium.org>
* Jan Woetzel
* Julien Schueller
* Kelly Thompson <kgt@lanl.gov>
* Laurent Montel <montel@kde.org>
* Konstantin Podsvirov <konstantin@podsvirov.pro>
* Mario Bensi <mbensi@ipsquad.net>
* Martin Gräßlin <mgraesslin@kde.org>
* Mathieu Malaterre <mathieu.malaterre@gmail.com>
* Matthaeus G. Chajdas
* Matthias Kretz <kretz@kde.org>
* Matthias Maennich <matthias@maennich.net>
* Michael Hirsch, Ph.D. <www.scivision.co>
* Michael Stürmer
* Miguel A. Figueroa-Villanueva
* Mike Jackson
* Mike McQuaid <mike@mikemcquaid.com>
* Nicolas Bock <nicolasbock@gmail.com>
* Nicolas Despres <nicolas.despres@gmail.com>
* Nikita Krupen'ko <krnekit@gmail.com>
* NVIDIA Corporation <www.nvidia.com>
* OpenGamma Ltd. <opengamma.com>
* Patrick Stotko <stotko@cs.uni-bonn.de>
* Per Øyvind Karlsen <peroyvind@mandriva.org>
* Peter Collingbourne <peter@pcc.me.uk>
* Petr Gotthard <gotthard@honeywell.com>
* Philip Lowman <philip@yhbt.com>
* Philippe Proulx <pproulx@efficios.com>
* Raffi Enficiaud, Max Planck Society
* Raumfeld <raumfeld.com>
* Roger Leigh <rleigh@codelibre.net>
* Rolf Eike Beer <eike@sf-mail.de>
* Roman Donchenko <roman.donchenko@itseez.com>
* Roman Kharitonov <roman.kharitonov@itseez.com>
* Ruslan Baratov
* Sebastian Holtermann <sebholt@xwmw.org>
* Stephen Kelly <steveire@gmail.com>
* Sylvain Joubert <joubert.sy@gmail.com>
* Thomas Sondergaard <ts@medical-insight.com>
* Tobias Hunger <tobias.hunger@qt.io>
* Todd Gamblin <tgamblin@llnl.gov>
* Tristan Carel
* University of Dundee
* Vadim Zhukov
* Will Dicharry <wdicharry@stellarscience.com>

See version control history for details of individual contributions.

The above copyright and license notice applies to distributions of
CMake in source and binary form.  Third-party software packages supplied
with CMake under compatible licenses provide their own copyright notices
documented in corresponding subdirectories or source files.

------------------------------------------------------------------------------

CMake was initially developed by Kitware with the following sponsorship:

 * National Library of Medicine at the National Institutes of Health
   as part of the Insight Segmentation and Registration Toolkit (ITK).

 * US National Labs (Los Alamos, Livermore, Sandia) ASC Parallel
   Visualization Initiative.

 * National Alliance for Medical Image Computing (NAMIC) is funded by the
   National Institutes of Health through the NIH Roadmap for Medical Research,
   Grant U54 EB005149.

 * Kitware, Inc.
#]]

# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindGDAL
--------

Find GDAL.

IMPORTED Targets
^^^^^^^^^^^^^^^^

This module defines :prop_tgt:`IMPORTED` target ``GDAL::GDAL``
if GDAL has been found.

Result Variables
^^^^^^^^^^^^^^^^

This module will set the following variables in your project:

``GDAL_FOUND``
  True if GDAL is found.
``GDAL_INCLUDE_DIRS``
  Include directories for GDAL headers.
``GDAL_LIBRARIES``
  Libraries to link to GDAL.
``GDAL_VERSION``
  The version of GDAL found.

Cache variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``GDAL_LIBRARY``
  The libgdal library file.
``GDAL_INCLUDE_DIR``
  The directory containing ``gdal.h``.

Hints
^^^^^

Set ``GDAL_DIR`` or ``GDAL_ROOT`` in the environment to specify the
GDAL installation prefix.
#]=======================================================================]

# $GDALDIR is an environment variable that would
# correspond to the ./configure --prefix=$GDAL_DIR
# used in building gdal.
#
# Created by Eric Wing. I'm not a gdal user, but OpenSceneGraph uses it
# for osgTerrain so I whipped this module together for completeness.
# I actually don't know the conventions or where files are typically
# placed in distros.
# Any real gdal users are encouraged to correct this (but please don't
# break the OS X framework stuff when doing so which is what usually seems
# to happen).

# This makes the presumption that you are include gdal.h like
#
#include "gdal.h"

find_path(GDAL_INCLUDE_DIR gdal.h
  HINTS
    ENV GDAL_DIR
    ENV GDAL_ROOT
  PATH_SUFFIXES
     include/gdal
     include/GDAL
     include
)

if(UNIX)
    # Use gdal-config to obtain the library version (this should hopefully
    # allow us to -lgdal1.x.y where x.y are correct version)
    # For some reason, libgdal development packages do not contain
    # libgdal.so...
    find_program(GDAL_CONFIG gdal-config
        HINTS
          ENV GDAL_DIR
          ENV GDAL_ROOT
        PATH_SUFFIXES bin
    )

    if(GDAL_CONFIG)
        exec_program(${GDAL_CONFIG} ARGS --libs OUTPUT_VARIABLE GDAL_CONFIG_LIBS)

        if(GDAL_CONFIG_LIBS)
            # treat the output as a command line and split it up
            separate_arguments(args NATIVE_COMMAND "${GDAL_CONFIG_LIBS}")

            # only consider libraries whose name matches this pattern
            set(name_pattern "[gG][dD][aA][lL]")

            # consider each entry as a possible library path, name, or parent directory
            foreach(arg IN LISTS args)
                # library name
                if("${arg}" MATCHES "^-l(.*)$")
                    set(lib "${CMAKE_MATCH_1}")

                    # only consider libraries whose name matches the expected pattern
                    if("${lib}" MATCHES "${name_pattern}")
                        list(APPEND _gdal_lib "${lib}")
                    endif()
                # library search path
                elseif("${arg}" MATCHES "^-L(.*)$")
                    list(APPEND _gdal_libpath "${CMAKE_MATCH_1}")
                # assume this is a full path to a library
                elseif(IS_ABSOLUTE "${arg}" AND EXISTS "${arg}")
                    # extract the file name
                    get_filename_component(lib "${arg}" NAME)

                    # only consider libraries whose name matches the expected pattern
                    if(NOT "${lib}" MATCHES "${name_pattern}")
                        continue()
                    endif()

                    # extract the file directory
                    get_filename_component(dir "${arg}" DIRECTORY)

                    # remove library prefixes/suffixes
                    string(REGEX REPLACE "^(${CMAKE_SHARED_LIBRARY_PREFIX}|${CMAKE_STATIC_LIBRARY_PREFIX})" "" lib "${lib}")
                    string(REGEX REPLACE "(${CMAKE_SHARED_LIBRARY_SUFFIX}|${CMAKE_STATIC_LIBRARY_SUFFIX})$" "" lib "${lib}")

                    # use the file name and directory as hints
                    list(APPEND _gdal_libpath "${dir}")
                    list(APPEND _gdal_lib "${lib}")
                endif()
            endforeach()
        endif()
    endif()
endif()

find_library(GDAL_LIBRARY
  NAMES ${_gdal_lib} gdal gdal_i gdal1.5.0 gdal1.4.0 gdal1.3.2 GDAL
  HINTS
     ENV GDAL_DIR
     ENV GDAL_ROOT
     ${_gdal_libpath}
  PATH_SUFFIXES lib
)

if (EXISTS "${GDAL_INCLUDE_DIR}/gdal_version.h")
    file(STRINGS "${GDAL_INCLUDE_DIR}/gdal_version.h" _gdal_version
        REGEX "GDAL_RELEASE_NAME")
    string(REGEX REPLACE ".*\"\(.*\)\"" "\\1" GDAL_VERSION "${_gdal_version}")
    unset(_gdal_version)
else ()
    set(GDAL_VERSION GDAL_VERSION-NOTFOUND)
endif ()

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GDAL
    VERSION_VAR GDAL_VERSION
    REQUIRED_VARS GDAL_LIBRARY GDAL_INCLUDE_DIR)

if (GDAL_FOUND AND NOT TARGET GDAL::GDAL)
    add_library(GDAL::GDAL UNKNOWN IMPORTED)
    set_target_properties(GDAL::GDAL PROPERTIES
        IMPORTED_LOCATION "${GDAL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${GDAL_INCLUDE_DIR}")
endif ()

set(GDAL_LIBRARIES ${GDAL_LIBRARY})
set(GDAL_INCLUDE_DIRS ${GDAL_INCLUDE_DIR})
