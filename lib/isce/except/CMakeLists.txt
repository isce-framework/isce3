####Set the source files
set(CPPS Error.cpp)

#####Library headers
set(HEADERS Error.h)

add_isce_libdir(except "${CPPS}" "${HEADERS}")
