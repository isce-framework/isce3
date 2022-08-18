#include "SubSwaths.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using isce3::product::SubSwaths;

void addbinding(pybind11::class_<SubSwaths>& pySubSwaths)
{
    pySubSwaths
            .def(py::init<>())
            .def(py::init<const int, const int, const int>(),
                 py::arg("length"), py::arg("width"),
                 py::arg("num_sub_swaths") = 1)
            .def(py::init<const int, const int,
                 std::vector<isce3::core::Matrix<int>>&>(),
                 py::arg("length"), py::arg("width"), 
                 py::arg("valid_samples_arrays_vect"))
            .def_property("num_sub_swaths",
                    py::overload_cast<>(
                            &SubSwaths::numSubSwaths, py::const_),
                    py::overload_cast<int>(&SubSwaths::numSubSwaths), R"(
                Number of sub-swaths
                )")
            .def("get_valid_samples_array",
                    &SubSwaths::getValidSamplesArray,
                    py::arg("sub_swath_number"), R"(
                Get valid samples for a sub-swath's array indexed from 1 (1st sub-swath)
                )")
            .def("set_valid_samples_array",
                    &SubSwaths::setValidSamplesArray,
                    py::arg("sub_swath_number"), R"(
                Set valid samples for a sub-swath's array indexed from 1 (1st sub-swath) 
                )",
                    py::arg("valid_samples_array"))
            .def("get_valid_samples_arrays_vect",
                    &SubSwaths::getValidSamplesArraysVect, R"(
                Get valid samples sub-swaths vector of arrays
                )")
            .def("set_valid_samples_arrays_vect",
                    &SubSwaths::setValidSamplesArraysVect,
                    py::arg("valid_samples_arrays_vect"), R"(
                Set valid samples sub-swaths vector of arrays
                )")
            .def("get_sample_sub_swath",
                    &SubSwaths::getSampleSubSwath,
                    py::arg("azimuth_index"), py::arg("range_index"), R"(
                Test if a radar sample belongs to a sub-swath or if it is invalid.
     
                Returns the 1-based index of the subswath that contains the pixel
                indexed by `azimuth_index` and `range_index`. If the pixel was not
                a member of any subswath, returns 0.

                If the dataset does not have sub-swaths valid-samples metadata, the
                dataset is considered to have a single sub-swath and all samples are
                treated as belonging to that first sub-swath. If sub-swath
                valid-samples are not provided for an existing sub-swath `s`
                (i.e. that subswath vector is empty), all samples of that sub-swath `s`
                will be considered valid. If more than one sub-swath has no sub-swaths
                valid-samples information, only the first sub-swath without
                valid-samples information will be returned. If the index of the first valid
                range pixel is greater than the index of the last valid range pixel,
                it is considered that the azimuth line does not have valid samples.
                )")
            .def("__getitem__", [](const SubSwaths& self, py::tuple key) {
                if (key.size() != 2) {
                        throw std::invalid_argument("require tuple of 2 numbers");
                }
                return self.getSampleSubSwath(key[0].cast<int>(), key[1].cast<int>());
            }, R"(
                Test if a radar sample belongs to a sub-swath or if it is invalid.
     
                Returns the 1-based index of the subswath that contains the pixel
                indexed by `azimuth_index` and `range_index`. If the pixel was not
                a member of any subswath, returns 0.

                If the dataset does not have sub-swaths valid-samples metadata, the
                dataset is considered to have a single sub-swath and all samples are
                treated as belonging to that first sub-swath. If sub-swath
                valid-samples are not provided for an existing sub-swath `s`
                (i.e. that subswath vector is empty), all samples of that sub-swath `s`
                will be considered valid. If more than one sub-swath has no sub-swaths
                valid-samples information, only the first sub-swath without
                valid-samples information will be returned. If the index of the first valid
                range pixel is greater than the index of the last valid range pixel,
                it is considered that the azimuth line does not have valid samples.
                )")
            .def_property("width",
                    py::overload_cast<>(&SubSwaths::width, py::const_),
                    py::overload_cast<const int&>(&SubSwaths::width), R"(
                Radar grid width
                )")
            .def_property("length",
                    py::overload_cast<>(&SubSwaths::length, py::const_),
                    py::overload_cast<const int&>(&SubSwaths::length), R"(
                Radar grid length
                )");
}
