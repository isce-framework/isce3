####Set the source files
set(SRCS
    ResampSlc.cpp)

#####Library headers
set(HEADERS ResampSlc.h
            ResampSlc.icc
            Tile.h
            Tile.icc)

add_isce_libdir(image "${SRCS}" "${HEADERS}")
