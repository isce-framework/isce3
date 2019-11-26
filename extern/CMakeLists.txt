add_subdirectory(googletest)
if(NOT HAVE_PYRE)
    add_subdirectory(pyre)
endif()

install(DIRECTORY cereal/include/cereal DESTINATION include
        FILES_MATCHING PATTERN *.hpp PATTERN *.h)
add_library(cereal INTERFACE)
target_include_directories(cereal SYSTEM INTERFACE cereal/include)

add_library(eigen INTERFACE)
target_include_directories(eigen SYSTEM INTERFACE eigen3)
