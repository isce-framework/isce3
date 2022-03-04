#include "math.h"
#include "Stats.h"

void addsubmodule_math(py::module & m)
{
    py::module m_math = m.def_submodule("math");

    // forward declare bound classes
    py::class_<isce3::math::Stats<float>>
        pyStatsFloat32(m_math, "StatsFloat32");
    py::class_<isce3::math::Stats<double>>
        pyStatsFloat64(m_math, "StatsFloat64");
    py::class_<isce3::math::Stats<std::complex<float>>>
        pyStatsCFloat32(m_math, "StatsCFloat32");
    py::class_<isce3::math::Stats<std::complex<double>>>
        pyStatsCFloat64(m_math, "StatsCFloat64");

    addbinding(pyStatsFloat32);
    addbinding(pyStatsFloat64);
    addbinding(pyStatsCFloat32);
    addbinding(pyStatsCFloat64);

    // forward declare bound classes
    addbinding_stats<float>(m_math, "compute_raster_stats_float32");
    addbinding_stats<double>(m_math, "compute_raster_stats_float64");
    addbinding_stats<std::complex<float>>(m_math, "compute_raster_stats_cfloat32");
    addbinding_stats<std::complex<double>>(m_math, "compute_raster_stats_cfloat64");

    py::class_<isce3::math::StatsRealImag<float>>
        pyStatsRealImagFloat32(m_math, "StatsRealImagFloat32");
    py::class_<isce3::math::StatsRealImag<double>>
        pyStatsRealImagFloat64(m_math, "StatsRealImagFloat64");

    addbinding(pyStatsRealImagFloat32);
    addbinding(pyStatsRealImagFloat64);

    addbinding_stats_real_imag<float>(m_math);
    addbinding_stats_real_imag<double>(m_math);
}
