set (PXDS Basis.pxd
          Cartesian.pxd
          DateTime.pxd
          Doppler.pxd
          Ellipsoid.pxd
          EulerAngles.pxd
          Interpolator.pxd
          LinAlg.pxd
          LUT1d.pxd
          LUT2d.pxd
          Matrix.pxd
          Orbit.pxd
          Peg.pxd
          Pegtrans.pxd
          Poly1d.pxd
          Poly2d.pxd
          Position.pxd
          pyBasis.pxd
          pyDateTime.pxd
          pyDoppler.pxd
          pyEllipsoid.pxd
          pyEulerAngles.pxd
          pyInterpolator.pxd
          pyLinAlg.pxd
          pyLUT1d.pxd
          pyLUT2d.pxd
          pyOrbit.pxd
          pyPeg.pxd
          pyPegtrans.pxd
          pyPoly1d.pxd
          pyPoly2d.pxd
          pyPosition.pxd
          pyQuaternion.pxd
          pyTimeDelta.pxd
          Quaternion.pxd
          TimeDelta.pxd)

add_isce_cythondir(core "${PXDS}")

