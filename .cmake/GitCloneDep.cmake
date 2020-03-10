function(git_clone_dep url user repo tag)
    if(NOT Git_FOUND)
        message(FATAL_ERROR "Cannot clone package from github"
                            " - could not find `git` executable"
            )
    endif()

    if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/${repo}-src)
        find_package(Git)
        if(NOT GIT_FOUND)
            message(FATAL_ERROR "need git installed")
        endif()

        execute_process(
            COMMAND ${GIT_EXECUTABLE} clone --single-branch --depth 1 --branch
                    ${tag} https://${url}/${user}/${repo} ${repo}-src
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            RESULT_VARIABLE git_clone_result
            )
        if(NOT git_clone_result EQUAL "0")
            message(FATAL_ERROR "git clone failed with ${git_clone_result}")
        endif()
    endif()

    add_subdirectory(
        ${CMAKE_CURRENT_BINARY_DIR}/${repo}-src
        ${CMAKE_CURRENT_BINARY_DIR}/${repo}-bld
        )
endfunction()
