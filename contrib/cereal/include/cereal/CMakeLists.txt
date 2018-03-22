
add_subdirectory(archives)
add_subdirectory(details)
add_subdirectory(external)
add_subdirectory(types)

# Individual headers
set(EXPORT_HEADERS macros.hpp cereal.hpp access.hpp)
install(FILES ${EXPORT_HEADERS} DESTINATION ${CEREAL_INCLUDEDIR} COMPONENT cereal_headers)

