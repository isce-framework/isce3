add_subdirectory(googletest)
if(NOT HAVE_PYRE)
    add_subdirectory(pyre)
endif()

install(DIRECTORY cereal/include/cereal DESTINATION include
        FILES_MATCHING PATTERN *.hpp PATTERN *.h)
add_library(cereal INTERFACE)
add_library(Cereal::Cereal ALIAS cereal)
target_include_directories(cereal SYSTEM INTERFACE cereal/include)

install(DIRECTORY eigen3/Eigen DESTINATION include)
add_library(eigen INTERFACE)
add_library(Eigen::Eigen ALIAS eigen)
target_include_directories(eigen SYSTEM INTERFACE eigen3)
