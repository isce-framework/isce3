
# Individual headers
set(EXPORT_HEADERS
    xml.hpp
    portable_binary.hpp
    json.hpp
    binary.hpp
    adapters.hpp)

install(FILES ${EXPORT_HEADERS} DESTINATION ${CEREAL_INCLUDEDIR}/archives COMPONENT cereal_headers)

