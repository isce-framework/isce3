####List the source files
set(SRCS
    Grass.cpp
    Neutron.cpp
    PhaseGrad.cpp
    Residue.cpp
    Tree.cpp
    Unwrap.cpp)

#####Library headers
set(HEADERS
    ICU.h
    ICU.icc
    LabelMap.h
    LabelMap.icc)

add_isce_libdir(unwrap/icu "${SRCS}" "${HEADERS}")
