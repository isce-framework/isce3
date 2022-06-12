#include "Stats.h"
#include <isce3/io/Raster.h>

#include <complex>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

namespace py = pybind11;

using isce3::math::Stats;
using isce3::math::StatsRealImag;

template<typename T>
void addbinding(py::class_<Stats<T>>& pyStats)
{
    pyStats.def(py::init<>());
    pyStats.def_readonly("min", &Stats<T>::min);
    pyStats.def_readonly("max", &Stats<T>::max);
    pyStats.def_readonly("mean", &Stats<T>::mean);
    pyStats.def_readonly("sample_stddev", &Stats<T>::sample_stddev);
    pyStats.def_readonly("n_valid", &Stats<T>::n_valid);
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
    pyStatsRealImag.def_readonly("min_real", &StatsRealImag<T>::min_real);
    pyStatsRealImag.def_readonly("max_real", &StatsRealImag<T>::max_real);
    pyStatsRealImag.def_readonly("mean_real", &StatsRealImag<T>::mean_real);
    pyStatsRealImag.def_readonly("sample_stddev_real", &StatsRealImag<T>::sample_stddev_real);
    pyStatsRealImag.def_readonly("min_imag", &StatsRealImag<T>::min_imag);
    pyStatsRealImag.def_readonly("max_imag", &StatsRealImag<T>::max_imag);
    pyStatsRealImag.def_readonly("mean_imag", &StatsRealImag<T>::mean_imag);
    pyStatsRealImag.def_readonly("sample_stddev_imag", &StatsRealImag<T>::sample_stddev_imag);
    pyStatsRealImag.def_readonly("n_valid", &StatsRealImag<T>::n_valid);
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


