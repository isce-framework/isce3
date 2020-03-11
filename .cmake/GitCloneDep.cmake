function(git_clone_dep url user repo tag)

    set(opts)
    set(onevalue REV PATCH)
    set(multivalue)
    cmake_parse_arguments(GCD "${opts}" "${onevalue}" "${multivalue}" ${ARGN})

    if(NOT Git_FOUND)
        message(FATAL_ERROR "Cannot clone package from github"
                            " - could not find `git` executable"
            )
    endif()

    # Helper macro to check subprocess return value
    macro(execute_process_checked)
        execute_process(${ARGN} RESULT_VARIABLE exec_result)
        if(NOT exec_result EQUAL "0")
            message(FATAL_ERROR "execute_process (called with '${ARGN}')"
                                " failed with ${exec_result}"
                )
        endif()
    endmacro()

    if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/${repo}-src)
        find_package(Git)
        if(NOT GIT_FOUND)
            message(FATAL_ERROR "need git installed")
        endif()

        # Shallow clone if we don't need to checkout a specific rev
        set(args)
        if(NOT DEFINED GCD_REV)
            set(args --depth 1)
        endif()
        execute_process_checked(
            COMMAND ${GIT_EXECUTABLE} clone --single-branch --branch ${tag}
                    https://${url}/${user}/${repo} ${repo}-src ${args}
            WORKING_DIRECTORY
            ${CMAKE_CURRENT_BINARY_DIR}
            )
        if(DEFINED GCD_REV)
            execute_process_checked(
                COMMAND ${GIT_EXECUTABLE} checkout ${GCD_REV} WORKING_DIRECTORY
                ${CMAKE_CURRENT_BINARY_DIR}/${repo}-src
                )
        endif()
        if(DEFINED GCD_PATCH)
            execute_process_checked(
                COMMAND ${GIT_EXECUTABLE} apply
                ${CMAKE_CURRENT_SOURCE_DIR}/${GCD_PATCH} WORKING_DIRECTORY
                ${CMAKE_CURRENT_BINARY_DIR}/${repo}-src
                )
        endif()
    endif()

    add_subdirectory(
        ${CMAKE_CURRENT_BINARY_DIR}/${repo}-src
        ${CMAKE_CURRENT_BINARY_DIR}/${repo}-bld
        )
endfunction()
