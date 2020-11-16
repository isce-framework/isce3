# based on https://github.com/lefticus/cppbestpractices

function(set_warnings target)
    option(WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
    if(WARNINGS_AS_ERRORS)
        set(maybe_werror -Werror)
    endif()

    set(CXX_WARNINGS
        ${maybe_werror}
        -Wall
        -Wextra # reasonable and standard
        -Wshadow # warn the user if a variable declaration shadows one from a
                 # parent context
        -Wnon-virtual-dtor # warn the user if a class with virtual functions has
                           # a non-virtual destructor. This helps catch hard to
                           # track down memory errors
        -Wold-style-cast # warn for c-style casts
        -Wcast-align # warn for potential performance problem casts
        -Wunused # warn on anything being unused
        -Woverloaded-virtual # warn if you overload (not override) a virtual
                             # function
        -Wpedantic # warn if non-standard C++ is used
        -Wconversion # warn on type conversions that may lose data
        -Wsign-conversion # warn on sign conversions
        -Wdouble-promotion # warn if float is implicit promoted to double
        -Wformat=2 # warn on security issues around functions that format output
                   # (ie printf)
        -Werror=switch # turn warnings controlled by -Wswitch into errors
        )

    set(CLANG_WARNINGS "")

    set(GCC_WARNINGS
        -Wnull-dereference # warn if a null dereference is detected
        -Wmisleading-indentation # warn if identation implies blocks where
                                 # blocks do not exist
        -Wduplicated-cond # warn if if / else chain has duplicated conditions
        -Wlogical-op # warn about logical operations being used where bitwise
                     # were probably wanted
        )

    # These should be fixed eventually but currently spew output
    list(APPEND CXX_WARNINGS
        -Wno-conversion
        -Wno-sign-conversion
        -Wno-float-conversion
        -Wno-double-promotion
        -Wno-sign-compare
        -Wno-old-style-cast
        -Wno-shadow
        )

    set(CUDA_WARNINGS
        ${CXX_WARNINGS}
        -Wno-pedantic
        )

    list(APPEND GCC_WARNINGS
        -Wno-useless-cast # warn if you perform a cast to the same type
        )
    list(APPEND CLANG_WARNINGS
        -Wno-mismatched-tags
        -Wno-shorten-64-to-32
        -Wno-implicit-int-conversion
        -Wno-implicit-float-conversion
        )

    # Use compiler-specific warnings
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(APPEND CXX_WARNINGS ${CLANG_WARNINGS})
    else()
        if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7)
            # warn if if / else branches have duplicated code
            #list(APPEND GCC_WARNINGS -Wduplicated-branches)
        endif()
        list(APPEND CXX_WARNINGS ${GCC_WARNINGS})
    endif()

    list(TRANSFORM CUDA_WARNINGS PREPEND -Xcompiler=)

    target_compile_options(${target} INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:  ${CXX_WARNINGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_WARNINGS}>
        )

endfunction()
