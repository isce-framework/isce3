###List of pyx and pxd for record
###Dont really need this here as these are already
###included in iscecore.pyx
set (CYISCE_PYX pyDateTime.pyx
                pyEllipsoid.pyx
                pyInterpolator.pyx
                pyLinAlg.pyx
                pyOrbit.pyx
                pyPeg.pyx
                pyPegTrans.pyx
                pyPoly1d.pyx
                pyPoly2d.pyx
                pyPosition.pyx)

set (CYISCE_PXD DateTime.pxd
                Ellipsoid.pxd
                Interpolator.pxd
                LinAlg.pxd
                Orbit.pxd
                Peg.pxd
                Pegtrans.pxd
                Poly1d.pxd
                Poly2d.pxd
                Position.pxd)

set (LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

set_source_files_properties(
    iscecore.pyx
    PROPERTIES CYTHON_IS_CXX TRUE)

cython_add_module(iscecore iscecore.pyx)
target_include_directories(iscecore PUBLIC ${ISCE_BUILDINCLUDEDIR} .)
target_link_libraries(iscecore ${LISCE} m)
install (TARGETS
        iscecore
        DESTINATION ${ISCE_PACKAGESDIR}
        COMPONENT extensions)
