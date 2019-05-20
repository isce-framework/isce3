set(PACKAGENAME core)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

add_subdirectory(interpolator)
add_subdirectory(lut)
add_subdirectory(orbit)
add_subdirectory(poly)
add_subdirectory(projections)
add_subdirectory(sinc2dinterpolator)
