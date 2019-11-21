
add_subdirectory(rapidxml)
add_subdirectory(rapidjson)

# Individual headers
set(EXPORT_HEADERS base64.hpp)
install(FILES ${EXPORT_HEADERS} DESTINATION ${CEREAL_INCLUDEDIR}/external COMPONENT cereal_headers)

