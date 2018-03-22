
# Individual headers
set(EXPORT_HEADERS error.h en.h)

install(FILES ${EXPORT_HEADERS} DESTINATION ${CEREAL_INCLUDEDIR}/external/rapidjson/error
    COMPONENT cereal_headers)

