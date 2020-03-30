include(FetchContent)

function(fetch_extern_repo repo)

    FetchContent_Declare(${repo} ${ARGN})

    FetchContent_GetProperties(${repo})
    if(NOT ${repo}_POPULATED)
        FetchContent_Populate(${repo})
        add_subdirectory(${${repo}_SOURCE_DIR} ${${repo}_BINARY_DIR})
    endif()
endfunction()
