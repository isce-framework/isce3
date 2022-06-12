#pragma once

#include "forward.h"

#include <pyre/journal.h>

namespace isce3 { namespace core {

// Default min and max block sizes in bytes per thread
constexpr static long long DEFAULT_MIN_BLOCK_SIZE = 1 << 25; // 32MB
constexpr static long long DEFAULT_MAX_BLOCK_SIZE = 1 << 28; // 256MB

/** Enumeration type to indicate memory management for processes
 * that require block processing in the Y direction
 */
enum class MemoryModeBlocksY {
    AutoBlocksY,     /**< auto mode (default value is defined by the
                         module that is being executed) */
    SingleBlockY,    /**< use a single block (disable block mode) */
    MultipleBlocksY, /**< use multiple blocks (enable block mode) */
};

/** Enumeration type to indicate memory management */
enum GeocodeMemoryMode {
    Auto = 0,                       /**< auto mode (default value is
                                         defined by the module that is
                                         being executed) */
    SingleBlock = 1,                /**< use a single block (disable
                                         block mode) */
    BlocksGeogrid = 2,              /**< use block processing only over the
                                         geogrid, i.e., load entire SLC at
                                         once and use it for all geogrid blocks */
    BlocksGeogridAndRadarGrid = 3   /**< use block processing over the
                                         geogrid and radargrid, i.e. the SLC is
                                         loaded in blocks for each geogrid block) */
};

// Get "human-readable" string of number of bytes 
std::string getNbytesStr(long long nbytes);

/** Compute the number of blocks and associated number of lines (length)
 * based on a minimum and maximum block size in bytes per thread
 *
 * @param[in]  array_length        Length of the data to be processed
 * @param[in]  array_width         Width of the data to be processed
 * @param[in]  nbands              Number of the bands to be processed
 * @param[in]  type_size           Type size of the data to be processed, in bytes
 * @param[in]  channel             Pyre info channel
 * @param[out] block_length        Block length
 * @param[out] nblock_y            Number of blocks in the Y direction
 * @param[in]  min_block_size      Minimum block size in bytes (per thread)
 * @param[in]  max_block_size      Maximum block size in bytes (per thread)
 * @param[in]  n_threads           Number of available threads (0 for auto)
 */
void getBlockProcessingParametersY(const int array_length, const int array_width,
        const int nbands = 1,
        const int type_size = 4, // Float32
        pyre::journal::info_t* channel = nullptr, int* block_length = nullptr,
        int* nblock_y = nullptr, 
        const long long min_block_size = DEFAULT_MIN_BLOCK_SIZE,
        const long long max_block_size = DEFAULT_MAX_BLOCK_SIZE,
        int n_threads = 0);

/** Compute the number of blocks and associated number of lines (length)
 * and columns (width) based on a minimum and maximum block size in bytes per thread.
 * If block_width` and `n_block_x` are both null, block division is only performed
 * in the Y direction.
 *
 * @param[in]  array_length        Length of the data to be processed
 * @param[in]  array_width         Width of the data to be processed
 * @param[in]  nbands              Number of the bands to be processed
 * @param[in]  type_size           Type size of the data to be processed, in bytes
 * @param[in]  channel             Pyre info channel
 * @param[out] block_length        Block length
 * @param[out] nblock_y            Number of blocks in the Y direction.
 * @param[out] block_width         Block width.
 *                                 If block_width` and `n_block_x` are both null,
 *                                 block division is only performed in the Y direction.
 * @param[out] nblock_x            Number of blocks in the X direction
 *  *                              If block_width` and `n_block_x` are both null,
 *                                 block division is only performed in the Y direction.
 * @param[in]  min_block_size      Minimum block size in bytes (per thread)
 * @param[in]  max_block_size      Maximum block size in bytes (per thread)
 * @param[in]  snap                Round block length and width to be multiples
 * of this value.
 * @param[in]  n_threads           Number of available threads (0 for auto)
 */
void getBlockProcessingParametersXY(const int array_length, const int array_width,
        const int nbands = 1,
        const int type_size = 4, // Float32
        pyre::journal::info_t* channel = nullptr,
        int* block_length = nullptr, int* nblock_y = nullptr,
        int* block_width = nullptr, int* nblock_x = nullptr,
        const long long min_block_size = DEFAULT_MIN_BLOCK_SIZE,
        const long long max_block_size = DEFAULT_MAX_BLOCK_SIZE,
        const int snap = 1, int n_threads = 0);

}}
