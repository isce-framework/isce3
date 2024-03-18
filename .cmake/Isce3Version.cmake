# Sets result variable in parent scope
# (Will be empty string on failure, or if the HEAD is a tagged commit)
function(isce3_get_git_hash result)

    if(DEFINED ${result})
        message(STATUS "Git rev already provided as '${${result}}'")
        return()
    endif()

    # Set to empty string so we can early exit on failure
    set(${result} "" PARENT_SCOPE)

    set(GIT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.git)
    if(NOT EXISTS ${GIT_DIR} OR NOT IS_DIRECTORY ${GIT_DIR})
        message(STATUS ".git directory does not exist")
        return()
    endif()

    find_package(Git)
    if(NOT Git_FOUND)
        message(STATUS "Could not find git program")
        return()
    endif()

    # Check if the current HEAD refers to a tagged commit.
    # Zero on success (HEAD is a tag); nonzero on failure (HEAD is not a tag)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --exact-match --tags HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE GIT_HEAD_IS_TAG
        )

    if(GIT_HEAD_IS_TAG EQUAL 0)
        message(STATUS "Current Git commit is tagged. Omitting hash from version string.")
        return()
    endif()

    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --always --dirty
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE GIT_REV_RESULT
        OUTPUT_VARIABLE GIT_REV_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    if(NOT GIT_REV_RESULT EQUAL 0)
        message(STATUS "git rev-parse failed (status ${GIT_REV_RESULT})")
        return()
    endif()

    set(${result} "${GIT_REV_OUTPUT}" PARENT_SCOPE)
endfunction()

# Sets version-components and description variables in parent scope
function(isce3_get_version components full)
    # Version should be a 3-component dotted version (e.g. "1.0.0")
    # followed by an optional hyphen and alphanumeric prerelease string
    set(VERSION_REGEX "([0-9]+\\.[0-9]+\\.[0-9]+)(-[a-zA-Z0-9]+)?")
    file(READ ${CMAKE_CURRENT_SOURCE_DIR}/VERSION.txt VERSION_FILE)
    string(STRIP "${VERSION_FILE}" VERSION_FILE)

    string(REGEX MATCH "${VERSION_REGEX}" VERSION_MATCHED "${VERSION_FILE}")
    if(NOT VERSION_FILE STREQUAL "${VERSION_MATCHED}")
        message(FATAL_ERROR "
            Invalid version string '${VERSION_FILE}'
            does not match '${VERSION_REGEX}'
        ")
    endif()
    set(VERSION_COMPONENTS "${CMAKE_MATCH_1}")
    set(VERSION_FULL "${VERSION_FILE}")

    # Append the git hash (if possible)
    isce3_get_git_hash(ISCE3_GIT_REV)
    if(ISCE3_GIT_REV)
        string(APPEND VERSION_FULL "+${ISCE3_GIT_REV}")
    endif()

    # Return values
    set(${components} "${VERSION_COMPONENTS}" PARENT_SCOPE)
    set(${full} "${VERSION_FULL}" PARENT_SCOPE)
endfunction()
