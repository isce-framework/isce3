
add_subdirectory(concepts)

# Individual headers
set(EXPORT_HEADERS
    array.hpp
    base_class.hpp
    bitset.hpp
    boost_variant.hpp
    chrono.hpp
    common.hpp
    complex.hpp
    deque.hpp
    forward_list.hpp
    functional.hpp
    list.hpp
    map.hpp
    memory.hpp
    polymorphic.hpp
    queue.hpp
    set.hpp
    stack.hpp
    string.hpp
    tuple.hpp
    unordered_map.hpp
    unordered_set.hpp
    utility.hpp
    valarray.hpp
    vector.hpp)

install(FILES ${EXPORT_HEADERS} DESTINATION ${CEREAL_INCLUDEDIR}/types COMPONENT cereal_headers)

