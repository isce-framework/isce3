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

void addbinding_get_block_processing_parameters(pybind11::module& m) {
    m.def("get_block_processing_parameters", [](const int array_length,
                                                const int array_width,
                                                const int nbands,
                                                const int type_size,
                                                const long long min_block_size,
                                                const long long max_block_size,
                                                const int snap,
                                                const int n_threads) {

                int block_length = 0;
                int block_width = 0;
                int n_blocks_y = 0;
                int n_blocks_x = 0;
                pyre::journal::info_t* info_channel = nullptr;

                isce3::core::getBlockProcessingParametersXY(
                        array_length, array_width,
                        nbands, type_size, info_channel,
                        &block_length, &n_blocks_y,
                        &block_width, &n_blocks_x,
                        min_block_size, max_block_size,
                        snap, n_threads);

                py::dict dict;
                dict["block_length"] = block_length;
                dict["block_width"] = block_width;
                dict["n_blocks_y"] = n_blocks_y;
                dict["n_blocks_x"] = n_blocks_x;
                return dict;
        },
         py::arg("array_length"),
         py::arg("array_width"),
         py::arg("nbands") = 1,
         py::arg("type_size") = 4,
         py::arg("min_block_size") = isce3::core::DEFAULT_MIN_BLOCK_SIZE,
         py::arg("max_block_size") = isce3::core::DEFAULT_MAX_BLOCK_SIZE,
         py::arg("snap") = 1,
         py::arg("n_threads") = 1,
         R"(
            Compute the number of blocks and associated number of lines
            (length) and columns (width) based on a minimum and maximum block
            size in bytes per thread.

            Parameters
            ----------
            array_length: int
                  Length of the data to be processed
            array_width: int
                  Width of the data to be processed
            nbands: int, optional
                  Number of the bands to be processed
            type_size: int, optional
                  Type size of the data to be processed, in bytes
            min_block_size: int, optional
                  Minimum block size in bytes (per thread)
            max_block_size: int, optional
                  Maximum block size in bytes (per thread)
            snap: int, optional
                  Round block length and width to be multiples of this value.
            n_threads: int, optional
                  Number of available threads (0 for auto)

            Returns
            -------
            ret_dict: dict
                Dictionary containing the following keys and values:
                * 'block_length': Block length
                * 'block_width': Block width.
                * 'n_block_y':  Number of blocks in the Y direction.
                * 'n_block_x': Number of blocks in the X direction. If
                   block_width` and `n_block_x` are both null, block division
                   is only performed in the Y direction.
        )");
}


void addbinding_get_block_processing_parameters_y(pybind11::module& m) {
    m.def("get_block_processing_parameters_y", [](const int array_length,
                                                  const int array_width,
                                                  const int nbands,
                                                  const int type_size,
                                                  const long long min_block_size,
                                                  const long long max_block_size,
                                                  const int n_threads) {

                int block_length = 0;
                int n_blocks_y = 0;
                pyre::journal::info_t* info_channel = nullptr;

                isce3::core::getBlockProcessingParametersY(
                        array_length, array_width,
                        nbands, type_size, info_channel,
                        &block_length, &n_blocks_y,
                        min_block_size, max_block_size,
                        n_threads);

                py::dict dict;
                dict["block_length"] = block_length;
                dict["n_blocks"] = n_blocks_y;

                return dict;
        },
         py::arg("array_length"),
         py::arg("array_width"),
         py::arg("nbands") = 1,
         py::arg("type_size") = 4,
         py::arg("min_block_size") = isce3::core::DEFAULT_MIN_BLOCK_SIZE,
         py::arg("max_block_size") = isce3::core::DEFAULT_MAX_BLOCK_SIZE,
         py::arg("n_threads") = 1,
         R"(
            Compute the number of blocks and associated number of lines
            (length) based on a minimum and maximum block size in bytes per
            thread

            Parameters
            ----------
            array_length: int
                  Length of the data to be processed
            array_width: int
                  Width of the data to be processed
            nbands: int, optional
                  Number of the bands to be processed
            type_size: int, optional
                  Type size of the data to be processed, in bytes
            min_block_size: int, optional
                  Minimum block size in bytes (per thread)
            max_block_size: int, optional
                  Maximum block size in bytes (per thread)
            n_threads: int, optional
                  Number of available threads (0 for auto)

            Returns
            -------
            ret_dict: dict
                Dictionary containing the following keys and values:
                * 'block_length': Block length
                * 'n_blocks':  Number of blocks in the Y direction.
        )");
}
