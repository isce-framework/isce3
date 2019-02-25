#####Library headers
set(HEADERS
    ComplexImagery.h
    Identification.h
    Identification.icc
    ImageMode.h
    ImageMode.icc
    Metadata.h
    Metadata.icc
    Product.h
    Product.icc
    Serialization.h
)

add_isce_libdir(product "${SRCS}" "${HEADERS}")
