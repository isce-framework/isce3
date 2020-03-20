include(FetchContent)

function(git_clone_dep url user repo)

    FetchContent_Declare(${repo}
        GIT_REPOSITORY https://${url}/${user}/${repo}.git
        GIT_SHALLOW TRUE ${ARGN})

    FetchContent_GetProperties(${repo})
    FetchContent_Populate(${repo})
    add_subdirectory(${${repo}_SOURCE_DIR} ${${repo}_BINARY_DIR})
endfunction()
