#include "symmetrize.h"

#include <isce3/core/DenseMatrix.h>
#include <isce3/math/complexOperations.h>

namespace isce3 { namespace polsar {

static void _validate_rasters(isce3::io::Raster& raster_a,
        std::string raster_a_name, int raster_a_band,
        isce3::io::Raster& raster_b, std::string raster_b_name,
        int raster_b_band)
{
    std::string error_msg;

    if (raster_a_band < 1 || raster_a_band > raster_a.numBands()) {
        error_msg = " Invalid band for " + raster_a_name + ": " +
                    std::to_string(raster_a_band);
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }
    if (raster_b_band < 1 || raster_b_band > raster_b.numBands()) {
        error_msg = " Invalid band for " + raster_b_name + ": " +
                    std::to_string(raster_b_band);
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }

    error_msg = "ERROR the ";
    error_msg += raster_a_name;
    error_msg += " and ";
    error_msg += raster_b_name;

    if (raster_a.length() != raster_b.length()) {
        error_msg += " raster dimensions to not match";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }
    if (raster_a.width() != raster_b.width()) {
        error_msg += " raster dimensions to not match";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }

    if (GDALDataTypeIsComplex(raster_a.dtype(raster_a_band)) xor
            GDALDataTypeIsComplex(raster_b.dtype(raster_b_band))) {
        error_msg += " raster data type to not match";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }
}

template<typename T>
void _symmetrizeCrossPolChannels(isce3::io::Raster& hv_raster,
        isce3::io::Raster& vh_raster, isce3::io::Raster& output_raster,
        const long x0, const long block_width, const long y0,
        const long block_length, const int hv_raster_band,
        const int vh_raster_band, const int output_raster_band)
{

    using namespace isce3::math::complex_operations;

    // Read HV array
    isce3::core::Matrix<T> hv_array(block_length, block_width);
    hv_raster.getBlock(
            hv_array.data(), x0, y0, block_width, block_length, hv_raster_band);

    // Read VH array
    isce3::core::Matrix<T> vh_array(block_length, block_width);
    vh_raster.getBlock(
            vh_array.data(), x0, y0, block_width, block_length, vh_raster_band);

    // Compute output
    isce3::core::Matrix<T> output_array(block_length, block_width);
    _Pragma("omp parallel for schedule(dynamic)") for (long i = 0;
                                                       i < block_length; ++i)
    {
        for (long j = 0; j < block_width; ++j) {
            output_array(i, j) = 0.5 * (hv_array(i, j) + vh_array(i, j));
        }
    }

    // Set output block
    output_raster.setBlock(output_array.data(), x0, y0, block_width,
            block_length, output_raster_band);
}

void symmetrizeCrossPolChannels(isce3::io::Raster& hv_raster,
        isce3::io::Raster& vh_raster, isce3::io::Raster& output_raster,
        isce3::core::MemoryModeBlocksY memory_mode, int hv_raster_band,
        int vh_raster_band, int output_raster_band)
{

    pyre::journal::info_t info("isce3.polsar.symmetrizeCrossPolChannels");

    info << "Symmetrizing cross-polarimetric channels (HV and VH)"
         << pyre::journal::endl;

    _validate_rasters(
            hv_raster, "HV", hv_raster_band, vh_raster, "VH", vh_raster_band);
    _validate_rasters(hv_raster, "HV", hv_raster_band, output_raster, "output",
            output_raster_band);

    const long x0 = 0;
    auto block_width = hv_raster.width();
    int block_length, nblocks;

    switch (memory_mode) {
    case isce3::core::MemoryModeBlocksY::SingleBlockY:
        nblocks = 1;
        block_length = static_cast<int>(hv_raster.length());
        break;
    case isce3::core::MemoryModeBlocksY::AutoBlocksY: [[fallthrough]];
    case isce3::core::MemoryModeBlocksY::MultipleBlocksY:
        const int out_nbands = 1;
        isce3::core::getBlockProcessingParametersY(hv_raster.length(),
                hv_raster.width(), out_nbands,
                GDALGetDataTypeSizeBytes(hv_raster.dtype()), &info,
                &block_length, &nblocks);
    }

    for (int block = 0; block < nblocks; ++block) {

        const long y0 = block * block_length;
        int this_block_length = block_length;
        if ((block + 1) * block_length > hv_raster.length()) {
            this_block_length = hv_raster.length() % block_length;
        }

        if (nblocks > 1) {
            info << "symmetrizing block: " << block + 1 << "/" << nblocks
                 << pyre::journal::endl;
        }

        if (hv_raster.dtype() == GDT_Float32)
            _symmetrizeCrossPolChannels<float>(hv_raster, vh_raster,
                    output_raster, x0, block_width, y0, this_block_length,
                    hv_raster_band, vh_raster_band, output_raster_band);
        else if (hv_raster.dtype() == GDT_Float64)
            _symmetrizeCrossPolChannels<double>(hv_raster, vh_raster,
                    output_raster, x0, block_width, y0, this_block_length,
                    hv_raster_band, vh_raster_band, output_raster_band);
        else if (hv_raster.dtype() == GDT_CFloat32)
            _symmetrizeCrossPolChannels<std::complex<float>>(hv_raster,
                    vh_raster, output_raster, x0, block_width, y0,
                    this_block_length, hv_raster_band, vh_raster_band,
                    output_raster_band);
        else if (hv_raster.dtype() == GDT_CFloat64)
            _symmetrizeCrossPolChannels<std::complex<double>>(hv_raster,
                    vh_raster, output_raster, x0, block_width, y0,
                    this_block_length, hv_raster_band, vh_raster_band,
                    output_raster_band);
        else {
            std::string error_message =
                    "ERROR not implemented for input raster datatype";
            throw isce3::except::RuntimeError(ISCE_SRCINFO(), error_message);
        }
    }
}
}} // namespace isce3::polsar
