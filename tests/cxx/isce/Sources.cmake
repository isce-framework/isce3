set(TESTFILES
container/rsd.cpp
core/attitude/euler.cpp
core/cube/cube.cpp
core/datetime/datetime.cpp
core/ellipsoid/ellipsoid.cpp
core/interp1d.cpp
core/interpolator/interpolator.cpp
core/linspace/linspace.cpp
core/lut/lut1d.cpp
core/lut/lut2d.cpp
core/matrix/matrix.cpp
core/orbit/orbit.cpp
core/poly/poly1d.cpp
core/poly/poly2d.cpp
core/projections/cea.cpp
core/projections/geocent.cpp
core/projections/polar.cpp
core/projections/utm.cpp
core/serialization/serializeAttitude.cpp
core/serialization/serializeDoppler.cpp
core/serialization/serializeOrbit.cpp
fft/fft.cpp
fft/fftplan.cpp
fft/fftutil.cpp
focus/bistatic-delay.cpp
focus/chirp.cpp
focus/dry-troposphere-model.cpp
focus/gaps.cpp
focus/rangecomp.cpp
geometry/dem/dem.cpp
geometry/geo2rdr/geo2rdr.cpp
geometry/geocode/geocode.cpp
geometry/geometry/geometry_constlat.cpp
geometry/geometry/geometry.cpp
geometry/geometry/geometry_equator.cpp
geometry/rtc/rtc.cpp
geometry/topo/topo.cpp
geometry/bbox/geoperimeter_equator.cpp
image/resampslc/resampslc.cpp
io/gdal/buffer.cpp
io/gdal/gdal-dataset.cpp
io/gdal/geotransform.cpp
io/gdal/spatialreference.cpp
io/IH5/ih5castread.cpp
io/IH5/ih5castwrite.cpp
io/IH5/ih5.cpp
io/IH5/ih5gdal.cpp
io/IH5/ih5nativeread.cpp
io/IH5/ih5nativewrite.cpp
io/raster/raster.cpp
io/raster/rasterepsg.cpp
io/raster/rastermatrix.cpp
io/raster/rasterview.cpp
matchtemplate/ampcor/ampcor.cpp
math/bessel/bessel53.cpp
math/sinc.cpp
product/serialization/serializeProduct.cpp
product/serialization/serializeProductMetadata.cpp
product/radargrid/radargrid.cpp
signal/covariance.cpp
signal/crossmul.cpp
signal/filter.cpp
signal/multilook.cpp
signal/nfft.cpp
signal/shift_signal.cpp
signal/signal.cpp
unwrap/icu/icu.cpp
unwrap/phass/phass.cpp
)

#This is a temporary fix - since GDAL does not support
#expose virtualmemmap on OS X. This will be revisited
#when GDAL adds the functionality or we update to add
#preprocessor directives to handle this.
if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    LIST(APPEND TESTFILES io/gdal/gdal-raster.cpp)
endif()
