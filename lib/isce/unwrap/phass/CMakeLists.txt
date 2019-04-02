####List the source files
set(SRCS
    ASSP.cc
    BMFS.cc
    CannyEdgeDetector.cc
    ChangeDetector.cc
    EdgeDetector.cc
    PhaseStatistics.cc
    PhassUnwrapper.cc
    Point.cc
    RegionMap.cc
    sort.cc
    Phass.cpp)

#####Library headers
set(HEADERS
    ASSP.h
    BMFS.h
    CannyEdgeDetector.h
    ChangeDetector.h
    DataPatch.h
    EdgeDetector.h
    PhaseStatistics.h
    PhassUnwrapper.h
    Point.h
    RegionMap.h
    constants.h
    sort.h
    Phass.h)

add_isce_libdir(unwrap/phass "${SRCS}" "${HEADERS}")
