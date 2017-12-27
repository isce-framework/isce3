#Mimicking mm behaviour
set(PACKAGENAME core)


####Build library
set(${PACKAGENAME}_SRCS Baseline.cpp
              DateTime.cpp
              Ellipsoid.cpp
              Interpolator.cpp
              LUT2d.cpp
              LinAlg.cpp
              Metadata.cpp
              Orbit.cpp
              Peg.cpp
              Pegtrans.cpp
              Poly1d.cpp
              Poly2d.cpp
              Position.cpp
              Projections.cpp)

set(${PACKAGENAME}_DLL isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

add_library(${${PACKAGENAME}_DLL} SHARED
                               ${${PACKAGENAME}_SRCS})

#Pyre stuff is needed for logging
target_include_directories(${${PACKAGENAME}_DLL} PUBLIC ${PYRE_INCLUDE_DIR}) 

#####Post building - installation
set(${PACKAGENAME}_EXPORT_HEADERS   Baseline.h
                                    Constants.h
                                    DateTime.h
                                    Ellipsoid.h
                                    Interpolator.h
                                    LUT2d.h
                                    LinAlg.h
                                    Metadata.h
                                    Orbit.h
                                    Peg.h
                                    Pegtrans.h
                                    Poly1d.h
                                    Poly2d.h
                                    Position.h
                                    Projections.h)

#Install headers as files
install(FILES ${${PACKAGENAME}_EXPORT_HEADERS}
    DESTINATION ${ISCE_INCLUDEDIR}/${PACKAGENAME}
    COMPONENT isce_headers)

#Install shared library
install(TARGETS ${${PACKAGENAME}_DLL}
        DESTINATION ${ISCE_LIBDIR}
        COMPONENT isce_libs)

