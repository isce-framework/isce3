####List the source files
set(SRCS
    Signal.cpp
    Crossmul.cpp
    Filter.cpp
    Looks.cpp
    Covariance.cpp)

####Append current path to every source file
#####Library headers
set(HEADERS
    Signal.h
    Signal.icc
    fftw3cxx.h
    Crossmul.h
    Crossmul.icc
    Filter.h
    Looks.h
    Looks.icc
    Covariance.h
    Covariance.icc)

add_isce_libdir(signal "${SRCS}" "${HEADERS}")
