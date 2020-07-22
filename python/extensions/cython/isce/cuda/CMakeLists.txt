set_source_files_properties(
    iscecudaextension.pyx
    PROPERTIES CYTHON_IS_CXX TRUE
)

cython_add_module(iscecudaextension iscecudaextension.pyx)
SetCoverageOptions(iscecudaextension)

# Override output shared object name to place in desired location
set_target_properties(iscecudaextension PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${ISCE_BUILDPACKAGESDIR}/isce3/extensions
)

target_include_directories(
        iscecudaextension PUBLIC
        ${PYTHON_NUMPY_INCLUDE_DIR}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(iscecudaextension PUBLIC ${LISCECUDA} ${LISCE} m)

install(
    TARGETS iscecudaextension
    DESTINATION ${ISCE_PACKAGESDIR}/isce3/extensions
    COMPONENT extensions
)
