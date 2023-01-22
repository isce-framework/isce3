#include "Stats.h"
#include <isce3/io/Raster.h>

#include <complex>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

using isce3::math::Stats;
using isce3::math::StatsRealImag;


std::string str_helper(const py::object obj,
                       const std::vector<std::string>& attrs)
{
    std::string out = py::str(obj.attr("__class__").attr("__name__"));
    out += "(";
    for (auto it = attrs.begin(); it != attrs.end(); ++it) {
        auto attr = *it;
        auto cattr = attr.c_str();
        out += attr + "=" + std::string(py::str(obj.attr(cattr)));
        if (it != attrs.end() - 1)
            out += ", ";
    }
    return out + ")";
}


template<typename T>
void addbinding(py::class_<Stats<T>>& pyStats)
{
    pyStats.def(py::init<>());
    pyStats.def_readonly("min", &Stats<T>::min);
    pyStats.def_readonly("max", &Stats<T>::max);
    pyStats.def_readonly("mean", &Stats<T>::mean);
    pyStats.def_property_readonly("sample_stddev", &Stats<T>::sample_stddev);
    pyStats.def_readonly("n_valid", &Stats<T>::n_valid);

    using ArrayT = py::array_t<T, py::array::c_style>;

    pyStats.def(py::init([](ArrayT x) {
        return Stats<T>(x.data(), x.size());
    }),
    "Calculate statistics of a block of data using Welford's algorithm.");

    pyStats.def("update", [](Stats<T>& self, ArrayT x) {
        const auto px = x.data();
        const size_t n = x.size();
        {
            py::gil_scoped_release release;
            self.update(px, n);
        }
    },
    R"(Calculate stats of a new block of data using Welford's algorithm and
    update current estimate with Chan's method.)");

    pyStats.def("update", [](Stats<T>& self, const Stats<T>& other) {
        return self.update(other);
    },
    "Update statistics with independent data using Chan's method");

    pyStats.def("__str__", [](const py::object self) {
        return str_helper(self,
            {"n_valid", "mean", "min", "max", "sample_stddev"});
    });
}

template void addbinding(py::class_<Stats<float>>&);
template void addbinding(py::class_<Stats<double>>&);
template void addbinding(py::class_<Stats<std::complex<float>>>&);
template void addbinding(py::class_<Stats<std::complex<double>>>&);

template<typename T>
void addbinding_stats(pybind11::module& m, const char * name)
{
    m.def(name, &isce3::math::computeRasterStats<T>,
          py::arg("input_raster"),
          py::arg("memory_mode") = isce3::core::MemoryModeBlocksY::AutoBlocksY,
          R"(Compute raster statistics.
 
             Calculate statistics (min, max, mean, and standard deviation) 
             from a multi-band raster.

             Parameters
             ----------
             input_raster : isce3.io.Raster
                 Input raster
             memory_mode : isce3.core.MemoryModeBlocksY
                 Memory mode

    )");
}

template void addbinding_stats<float>(py::module& m, const char* name);
template void addbinding_stats<double>(py::module& m, const char* name);
template void addbinding_stats<std::complex<float>>(py::module& m, const char* name);
template void addbinding_stats<std::complex<double>>(py::module& m, const char* name);

template<typename T>
void addbinding(py::class_<StatsRealImag<T>>& pyStatsRealImag)
{
    pyStatsRealImag.def(py::init<>());
    pyStatsRealImag.def_readonly("real", &StatsRealImag<T>::real);
    pyStatsRealImag.def_readonly("imag", &StatsRealImag<T>::imag);
    pyStatsRealImag.def_readonly("n_valid", &StatsRealImag<T>::n_valid);

    using ArrayT = py::array_t<std::complex<T>, py::array::c_style>;

    pyStatsRealImag.def(py::init([](ArrayT x) {
        return StatsRealImag<T>(x.data(), x.size());
    }),
    "Calculate statistics of a block of data using Welford's algorithm.");

    pyStatsRealImag.def("update", [](StatsRealImag<T>& self, ArrayT x) {
        const auto px = x.data();
        const size_t n = x.size();
        {
            py::gil_scoped_release release;
            self.update(px, n);
        }
    },
    R"(Calculate stats of a new block of data using Welford's algorithm and
    update current estimate with Chan's method.)");

    pyStatsRealImag.def("update", [](StatsRealImag<T>& self,
            const StatsRealImag<T>& other) {
        return self.update(other);
    });

    pyStatsRealImag.def("__str__", [](const py::object self) {
        return str_helper(self, {"real", "imag"});
    });
}

template void addbinding(py::class_<StatsRealImag<float>>&);
template void addbinding(py::class_<StatsRealImag<double>>&);

template<typename T>
void addbinding_stats_real_imag(pybind11::module& m)
{
    m.def("compute_raster_stats_real_imag", 
          &isce3::math::computeRasterStatsRealImag<T>,
          py::arg("input_raster"),
          py::arg("memory_mode") = isce3::core::MemoryModeBlocksY::AutoBlocksY,
          R"(Compute real and imaginary statistics separately from a
          complex-valued raster.
          
          Calculate real and imaginary statistics from a multi-band raster.

          Parameters
          ----------
          input_raster : isce3.io.Raster
              Input raster
          memory_mode : isce3.core.MemoryModeBlocksY
              Memory mode

    )");
}

template void addbinding_stats_real_imag<float>(py::module& m);
template void addbinding_stats_real_imag<double>(py::module& m);


