function(set_warnings target)

    set(CXX_WARNINGS "")

    option(ISCE3_WITH_WERROR "Treat all compiler warnings as errors" OFF)
    if(ISCE3_WITH_WERROR)
        list(APPEND CXX_WARNINGS -Werror)
    endif()

    set(CXX_CANDIDATE_WARNINGS
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
        -Werror=reorder # turn warnings controlled by -Wreorder into errors

        -Wnull-dereference # warn if a null dereference is detected
        -Wmisleading-indentation # warn if identation implies blocks where
                                 # blocks do not exist
        -Wduplicated-cond # warn if if / else chain has duplicated conditions
        -Wlogical-op # warn about logical operations being used where bitwise
                     # were probably wanted

        # These should be fixed eventually but currently spew output
        -Wno-conversion
        -Wno-sign-conversion
        -Wno-float-conversion
        -Wno-double-promotion
        -Wno-sign-compare
        -Wno-old-style-cast
        -Wno-shadow
        -Wno-useless-cast # warn if you perform a cast to the same type
        -Wno-mismatched-tags
        -Wno-shorten-64-to-32
        -Wno-implicit-int-conversion
        -Wno-implicit-float-conversion
        )

    set(langs CXX)
    if(CMAKE_CUDA_COMPILER)
        list(APPEND langs CUDA)
        set(CUDA_CANDIDATE_WARNINGS ${CXX_CANDIDATE_WARNINGS} -Wno-pedantic)
        if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
            list(TRANSFORM CUDA_CANDIDATE_WARNINGS PREPEND -Xcompiler=)
        endif()
    endif()

    # Check all the flags, and add any that are supported to the compile-line.
    include(CheckCompilerFlag)
    foreach(lang ${langs})
        foreach(warning ${${lang}_CANDIDATE_WARNINGS})
            check_compiler_flag(${lang} ${warning} ${lang}_FLAG_${warning})
            if(${lang}_FLAG_${warning})
                list(APPEND ${lang}_WARNINGS "${warning}")
            endif()
        endforeach()
        target_compile_options(${target} INTERFACE
            $<$<COMPILE_LANGUAGE:${lang}>:${${lang}_WARNINGS}>)
    endforeach()
endfunction()
