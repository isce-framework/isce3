
# Individual headers
set(EXPORT_HEADERS
    rapidxml.hpp
    rapidxml_iterators.hpp
    rapidxml_print.hpp
    rapidxml_utils.hpp)

install(FILES ${EXPORT_HEADERS} DESTINATION ${CEREAL_INCLUDEDIR}/external/rapidxml 
    COMPONENT cereal_headers)

