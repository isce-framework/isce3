#####Library headers
set(HEADERS
    forward.h
    RadarGridParameters.h
    Metadata.h
    ProcessingInformation.h
    Product.h
    Serialization.h
    Swath.h)

set(SRCS Product.cpp)

add_isce_libdir(product "${SRCS}" "${HEADERS}")
