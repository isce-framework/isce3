#
# Copyright 2017-2018
#

# Find Numpy
find_package(Numpy REQUIRED)

# Set the name for the ISCE C++ library
set(LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

# Set file properties
set_source_files_properties(
    isceextension.pyx
    PROPERTIES CYTHON_IS_CXX TRUE
)


###Start by configuring all pxds build/cyinclude
set(PXD_LIST    core/pyAttitude.pxd
	        core/pyBasis.pxd
	        core/pyDateTime.pxd
	        core/pyDoppler.pxd
	        core/pyEllipsoid.pxd
	        core/pyInterpolator.pxd
	        core/pyLinAlg.pxd
	        core/pyOrbit.pxd
	        core/pyPeg.pxd
	        core/pyPegtrans.pxd
	        core/pyPoly1d.pxd
	        core/pyPoly2d.pxd
	        core/pyPosition.pxd
	        core/pyTimeDelta.pxd
	        geometry/pyGeo2rdr.pxd
	        geometry/pyTopo.pxd
	        image/pyResampSlc.pxd
	        io/pyGDAL.pxd
	        io/pyIH5.pxd
	        io/pyRaster.pxd
	        product/pyComplexImagery.pxd
	        product/pyIdentification.pxd
	        product/pyImageMode.pxd
	        product/pyMetadata.pxd
	        product/pyProduct.pxd
	        radar/pyRadar.pxd) 

foreach(HFILE ${PXD_LIST})
    get_filename_component(HFILE_BASE ${HFILE} NAME)
    configure_file(
        ${CMAKE_CURRENT_LIST_DIR}/${HFILE}
        ${ISCE_BUILDCYINCLUDEDIR}/${HFILE_BASE}
        COPYONLY)
endforeach()


# Add Cython module
cython_add_module(isceextension isceextension.pyx)

# Add necessary include directories
target_include_directories(
        isceextension PUBLIC
        ${ISCE_BUILDINCLUDEDIR}
        ${PYTHON_NUMPY_INCLUDE_DIR}
        ${GDAL_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/contrib/cereal/include
        ${PYRE_INCLUDE_DIR}
)

# Add necessary library links
target_link_libraries(isceextension ${LISCE} m)

# Install
install(
    TARGETS isceextension
    DESTINATION ${ISCE_PACKAGESDIR}
    COMPONENT extensions
)

# Install cuda cython extensions if cuda available
if (WITH_CUDA)
    add_subdirectory(cuda)
endif(WITH_CUDA)

# end of file
