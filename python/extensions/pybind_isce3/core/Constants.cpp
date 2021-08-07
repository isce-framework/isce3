#include "Constants.h"
#include <isce3/core/Constants.h>

namespace py = pybind11;

using isce3::core::MemoryModeBlockY;

void add_constants(py::module & core)
{
    py::enum_<isce3::core::dataInterpMethod>(core, "DataInterpMethod")
        .value("SINC", isce3::core::SINC_METHOD)
        .value("BILINEAR", isce3::core::BILINEAR_METHOD)
        .value("BICUBIC", isce3::core::BICUBIC_METHOD)
        .value("NEAREST", isce3::core::NEAREST_METHOD)
        .value("BIQUINTIC", isce3::core::BIQUINTIC_METHOD);
        // nicer not to export_values() to parent namespace

    py::enum_<MemoryModeBlockY>(core, "memory_mode_block_y",
            R"(Enumeration type to indicate memory management for processes 
        that require block processing in the Y direction.)")
            .value("AutoBlocksY", MemoryModeBlockY::AutoBlocksY,
                    R"(auto mode (default value is defined by the module
                that is being executed))")
            .value("SingleBlockY", MemoryModeBlockY::SingleBlockY,
                    R"(use a single block (disable block mode))")
            .value("MultipleBlocksY", MemoryModeBlockY::MultipleBlocksY,
                    R"(use multiple blocks (enable block mode))");

    core.attr("speed_of_light") = py::float_(isce3::core::speed_of_light);
    core.attr("earth_spin_rate") = py::float_(isce3::core::EarthSpinRate);
}
