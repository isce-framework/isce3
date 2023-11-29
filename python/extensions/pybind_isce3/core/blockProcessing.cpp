#include "blockProcessing.h"
#include <isce3/core/blockProcessing.h>

namespace py = pybind11;

using isce3::core::GeocodeMemoryMode;
using isce3::core::MemoryModeBlocksY;


void addbinding_block_processing(py::module & core)
{
    py::enum_<GeocodeMemoryMode>(core, "GeocodeMemoryMode")
            .value("Auto", GeocodeMemoryMode::Auto,
                    R"(auto mode (default value is defined by the module that is
                      being executed.)")
            .value("SingleBlock", GeocodeMemoryMode::SingleBlock,
                    R"(use a single block (disable block mode))")
            .value("BlocksGeogrid", GeocodeMemoryMode::BlocksGeogrid,
                    R"(use block processing only over the geogrid, i.e., load
                      entire SLC at once and use it for all geogrid blocks)")
            .value("BlocksGeogridAndRadarGrid",
                   GeocodeMemoryMode::BlocksGeogridAndRadarGrid,
                    R"(use block processing over the geogrid and radargrid,
                      i.e. the SLC is loaded in blocks for each geogrid
                      block))");

    py::enum_<MemoryModeBlocksY>(core, "MemoryModeBlocksY",
                    R"(Enumeration type to indicate memory management for
                      processes that require block processing in the Y
                      direction.)")
            .value("AutoBlocksY", MemoryModeBlocksY::AutoBlocksY,
                    R"(auto mode (default value is defined by the module
                      that is being executed))")
            .value("SingleBlockY", MemoryModeBlocksY::SingleBlockY,
                    R"(use a single block (disable block mode))")
            .value("MultipleBlocksY", MemoryModeBlocksY::MultipleBlocksY,
                    R"(use multiple blocks (enable block mode))");

    core.attr("default_min_block_size") =
        isce3::core::DEFAULT_MIN_BLOCK_SIZE;
    core.attr("default_max_block_size") =
        isce3::core::DEFAULT_MAX_BLOCK_SIZE;
}
