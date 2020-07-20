#include "Covariance.h"

#include <isce3/io/Raster.h>

#include <pybind11/stl.h>

namespace py = pybind11;

using isce::signal::Covariance;

template<typename T>
void addbinding(py::class_<Covariance<T>> &pyCovariance)
{
    pyCovariance
        .def(py::init<>())
        .def("covariance", [](isce::signal::Covariance<T> & self,
                    std::map<std::string, isce::io::Raster> & slc,
                    std::map<std::pair<std::string, std::string>, isce::io::Raster> & cov,
                    size_t rng_looks, size_t az_looks)
            {
                // perform covariance
                self.covariance(slc, cov, rng_looks, az_looks);
            },
            py::arg("slc"),
            py::arg("cov"),
            py::arg("rng_looks")=1,
            py::arg("az_looks")=1)
    ;
}

template void addbinding(py::class_<Covariance<std::complex<float>>> &);
template void addbinding(py::class_<Covariance<std::complex<double>>> &);

// end of file
