
# Individual headers
set(EXPORT_HEADERS inttypes.h stdint.h)

install(FILES ${EXPORT_HEADERS} DESTINATION ${CEREAL_INCLUDEDIR}/external/rapidjson/msinttypes
    COMPONENT cereal_headers)

