
# Individual headers
set(EXPORT_HEADERS macros.hpp cereal.hpp access.hpp)
install(FILES ${EXPORT_HEADERS} DESTINATION include/cereal COMPONENT cereal_headers)

# Relevant directories
foreach(CEREAL_DIR archives details external types)
    install(DIRECTORY ${CEREAL_DIR} DESTINATION include/cereal COMPONENT cereal_headers)
endforeach()
