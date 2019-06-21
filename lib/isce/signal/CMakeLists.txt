####List the source files
set(SRCS
    Signal.cpp
    NFFT.cpp
    Crossmul.cpp
    Filter.cpp
    Looks.cpp
    Covariance.cpp
    shiftSignal.cpp)

####Append current path to every source file
#####Library headers
set(HEADERS
    Signal.h
    Signal.icc
    NFFT.h
    Crossmul.h
    Crossmul.icc
    Filter.h
    Looks.h
    Looks.icc
    Covariance.h
    Covariance.icc
    shiftSignal.h)

add_isce_libdir(signal "${SRCS}" "${HEADERS}")
