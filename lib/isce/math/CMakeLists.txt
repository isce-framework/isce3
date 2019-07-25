#Mimicking mm behaviour
set(PACKAGENAME math)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

####Set the source files
set(CPPS Bessel.cpp)

#####Library headers
set(HEADERS Bessel.h)

add_isce_libdir(math "${CPPS}" "${HEADERS}")
