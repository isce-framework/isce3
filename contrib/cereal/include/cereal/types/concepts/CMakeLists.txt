
# Individual headers
set(EXPORT_HEADERS pair_associative_container.hpp)
    
install(FILES ${EXPORT_HEADERS} DESTINATION ${CEREAL_INCLUDEDIR}/types/concepts
    COMPONENT cereal_headers)

