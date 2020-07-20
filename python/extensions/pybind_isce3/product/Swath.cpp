#include "Swath.h"

#include <string>
#include <valarray>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <isce3/io/IH5.h>
#include <isce3/product/Product.h>

namespace py = pybind11;

using isce::product::Swath;

void addbinding(pybind11::class_<Swath> & pySwath)
{
    pySwath
        .def(py::init<>())
        .def(py::init([](const std::string &h5file, const char freq)
                {
                    // open file
                    isce::io::IH5File file(h5file);

                    // instantiate and load a product
                    isce::product::Product product(file);

                    // return swath from product
                    return product.swath(freq);
                }),
                py::arg("h5file"), py::arg("freq"))
        .def_property_readonly("range_pixel_spacing", &Swath::rangePixelSpacing)
        .def_property_readonly("samples", &Swath::samples)
        .def_property_readonly("lines", &Swath::lines)
        .def_property_readonly("processed_wavelength", &Swath::processedWavelength)
        .def_property("slant_range",
                py::overload_cast<>(&Swath::slantRange, py::const_),
                py::overload_cast<const std::valarray<double> &>(&Swath::slantRange))
        .def_property("zero_doppler_time",
                py::overload_cast<>(&Swath::zeroDopplerTime, py::const_),
                py::overload_cast<const std::valarray<double> &>(&Swath::zeroDopplerTime))
        .def_property("acquired_center_frequency",
                py::overload_cast<>(&Swath::acquiredCenterFrequency, py::const_),
                py::overload_cast<double>(&Swath::acquiredCenterFrequency))
        .def_property("processed_center_frequency",
                py::overload_cast<>(&Swath::processedCenterFrequency, py::const_),
                py::overload_cast<double>(&Swath::processedCenterFrequency))
        .def_property("acquired_range_bandwidth",
                py::overload_cast<>(&Swath::acquiredRangeBandwidth, py::const_),
                py::overload_cast<double>(&Swath::acquiredRangeBandwidth))
        .def_property("processed_range_bandwidth",
                py::overload_cast<>(&Swath::processedRangeBandwidth, py::const_),
                py::overload_cast<double>(&Swath::processedRangeBandwidth))
        .def_property("nominal_acquisition_prf",
                py::overload_cast<>(&Swath::nominalAcquisitionPRF, py::const_),
                py::overload_cast<double>(&Swath::nominalAcquisitionPRF))
        .def_property("scene_center_along_track_spacing",
                py::overload_cast<>(&Swath::sceneCenterAlongTrackSpacing, py::const_),
                py::overload_cast<double>(&Swath::sceneCenterAlongTrackSpacing))
        .def_property("scene_center_ground_range_spacing",
                py::overload_cast<>(&Swath::sceneCenterGroundRangeSpacing, py::const_),
                py::overload_cast<double>(&Swath::sceneCenterGroundRangeSpacing))
        .def_property("processed_azimuth_bandwidth",
                py::overload_cast<>(&Swath::processedAzimuthBandwidth, py::const_),
                py::overload_cast<double>(&Swath::processedAzimuthBandwidth))
        .def_property("ref_epoch",
                py::overload_cast<>(&Swath::refEpoch, py::const_),
                py::overload_cast<const DateTime &>(&Swath::refEpoch))
        ;
}
