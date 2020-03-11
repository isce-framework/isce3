install(
    DIRECTORY cereal/include/cereal
    DESTINATION include
    FILES_MATCHING
    PATTERN *.hpp
    PATTERN *.h
    )
add_library(cereal INTERFACE)
add_library(Cereal::Cereal ALIAS cereal)
target_include_directories(
    cereal SYSTEM
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/cereal/include>
              $<INSTALL_INTERFACE:include>
    )

install(
    TARGETS cereal
    EXPORT isce3-targets
    DESTINATION ${ISCE_LIBDIR}
    )

add_subdirectory(pybind11)
