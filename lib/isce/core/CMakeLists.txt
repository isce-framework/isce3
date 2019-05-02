#Mimicking mm behaviour
set(PACKAGENAME core)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

####Set the source files
set(CPPS Attitude.cpp
         Baseline.cpp
         BilinearInterpolator.cpp
         BicubicInterpolator.cpp
         DateTime.cpp
         Doppler.cpp
         Ellipsoid.cpp
         EulerAngles.cpp
         Interp1d.cpp
         Interpolator.cpp
         Kernels.cpp
         LUT2d.cpp
         LinAlg.cpp
         Metadata.cpp
         NearestNeighborInterpolator.cpp
         Orbit.cpp
         Peg.cpp
         Pegtrans.cpp
         Poly1d.cpp
         Poly2d.cpp
         Position.cpp
         Projections.cpp
         Quaternion.cpp
         Sinc2dInterpolator.cpp
         Spline2dInterpolator.cpp
         TimeDelta.cpp)

#####Library headers
set(HEADERS Attitude.h
            Baseline.h
            Basis.h
            Constants.h
            DateTime.h
            Doppler.h
            Ellipsoid.h
            EulerAngles.h
            Interp1d.h
            Interpolator.h
            Kernels.h
            LUT1d.h
            LUT1d.icc
            LUT2d.h
            LinAlg.h
            Matrix.h
            Matrix.icc
            Metadata.h
            Orbit.h
            Peg.h
            Pegtrans.h
            Pixel.h
            Poly1d.h
            Poly2d.h
            Position.h
            Projections.h
            Quaternion.h
            Serialization.h
            Serialization.h
            StateVector.h
            TimeDelta.h
            Utilities.h)

add_isce_libdir(core "${CPPS}" "${HEADERS}")
