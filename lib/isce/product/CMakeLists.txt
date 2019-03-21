#####Library headers
set(HEADERS
    RadarGridParameters.h
    Metadata.h
    ProcessingInformation.h
    Product.h
    Serialization.h
    Swath.h)

add_isce_libdir(product "${SRCS}" "${HEADERS}")
