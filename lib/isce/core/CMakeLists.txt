#Mimicking mm behaviour
set(PACKAGENAME core)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

####Set the source files
set(CPPS Baseline.cpp
         BilinearInterpolator.cpp
         BicubicInterpolator.cpp
         DateTime.cpp
         Ellipsoid.cpp
         EulerAngles.cpp
         Interpolator.cpp
         Kernels.cpp
         LUT2d.cpp
         Metadata.cpp
         NearestNeighborInterpolator.cpp
         Orbit.cpp
         Peg.cpp
         Pegtrans.cpp
         Poly1d.cpp
         Poly2d.cpp
         Projections.cpp
         Quaternion.cpp
         Sinc2dInterpolator.cpp
         Spline2dInterpolator.cpp
         TimeDelta.cpp)

#####Library headers
set(HEADERS Attitude.h
            Baseline.h
            Basis.h
            Cartesian.h
            Common.h
            Constants.h
            Constants.h
            DateTime.h
            DenseMatrix.h
            Ellipsoid.h
            EulerAngles.h
            Interp1d.h
            Interp1d.icc
            Interpolator.h
            Kernels.h
            Linspace.h
            Linspace.icc
            LUT1d.h
            LUT1d.icc
            LUT2d.h
            Matrix.h
            Matrix.icc
            Cube.h
            Cube.icc
            Metadata.h
            Orbit.h
            Peg.h
            Pegtrans.h
            Pixel.h
            Poly1d.h
            Poly2d.h
            Projections.h
            Quaternion.h
            Serialization.h
            Serialization.h
            StateVector.h
            TimeDelta.h
            Utilities.h
            Vector.h)

add_isce_libdir(core "${CPPS}" "${HEADERS}")
