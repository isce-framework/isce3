
# Individual headers
set(EXPORT_HEADERS
    helpers.hpp
    polymorphic_impl.hpp
    polymorphic_impl_fwd.hpp
    static_object.hpp
    traits.hpp
    util.hpp)

install(FILES ${EXPORT_HEADERS} DESTINATION ${CEREAL_INCLUDEDIR}/details COMPONENT cereal_headers)

