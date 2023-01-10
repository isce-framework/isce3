#include "blockProcessing.h"

#include <cmath>
#include <algorithm>

#include <isce3/except/Error.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace isce3 { namespace core {

static int _omp_thread_count() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

std::string getNbytesStr(long long nbytes)
{
    std::string nbytes_str;
    if (nbytes < std::pow(2, 10))
        nbytes_str = std::to_string(nbytes) + "B";
    else if (nbytes < std::pow(2, 20))
        nbytes_str = std::to_string((int) std::ceil(nbytes / std::pow(2, 10))) +
                     "KB";
    else if (nbytes < std::pow(2, 30))
        nbytes_str = std::to_string((int) std::ceil(nbytes / std::pow(2, 20))) +
                     "MB";
    else
        nbytes_str = std::to_string((int) std::ceil(nbytes / std::pow(2, 30))) +
                     "GB";
    return nbytes_str;
}

void getBlockProcessingParametersY(const int array_length, const int array_width,
        const int nbands, const int type_size,
        pyre::journal::info_t* channel, int* block_length, int* nblocks_y,
        const long long min_block_size, const long long max_block_size,
        int n_threads)
{

    if (n_threads < 0) {
        std::string error_message = ("ERROR number of threads cannot be"
                                     " negative");
        throw isce3::except::OutOfRange(ISCE_SRCINFO(), error_message);
    }
    if (min_block_size > max_block_size) {
        std::string error_message = ("ERROR minimum block size cannot be"
                                     "greater than the maximum block size");
        throw isce3::except::RuntimeError(ISCE_SRCINFO(), error_message);
    }

    const int min_block_length = min_block_size /
        (static_cast<long long>(nbands) * array_width * type_size);
    const int max_block_length = max_block_size /
        (static_cast<long long>(nbands) * array_width * type_size);

    if (n_threads == 0) {
        n_threads = _omp_thread_count();
    }

    int _nblocks_y = std::max(n_threads, 1);
    int _block_length =
            std::ceil((static_cast<float>(array_length)) / _nblocks_y);
    _block_length = std::max(_block_length, min_block_length);
    _block_length = std::min({_block_length, max_block_length, array_length});

    // update nblocks_y
    _nblocks_y =
            std::ceil((static_cast<float>(array_length)) / _block_length);

    if (nblocks_y != nullptr)
        *nblocks_y = _nblocks_y;

    if (channel != nullptr) {
        *channel << "array length: " << array_length << pyre::journal::newline;
        *channel << "array width: " << array_width << pyre::journal::newline;
        *channel << "number of block(s): " << _nblocks_y
                 << pyre::journal::newline;
    }

    if (block_length != nullptr) {
        *block_length = _block_length;
        if (channel != nullptr) {
            *channel << "block length: " << *block_length
                     << pyre::journal::newline;
            *channel << "block width: " << array_width
                     << pyre::journal::newline;
        }
    }

    if (channel != nullptr) {

        long long block_size_bytes =
                ((static_cast<long long>(_block_length)) *
                 array_width * nbands * type_size);

        std::string block_size_bytes_str = isce3::core::getNbytesStr(block_size_bytes);
        if (nbands > 1)
            block_size_bytes_str += " (" + std::to_string(nbands) + " bands)";

        *channel << "block size: " << block_size_bytes_str
                 << pyre::journal::endl;
    }
}

void getBlockProcessingParametersXY(const int array_length, const int array_width,
        const int nbands, const int type_size, pyre::journal::info_t* channel,
        int* block_length, int* nblocks_y, int* block_width, int* nblocks_x,
        const long long min_block_size, const long long max_block_size,
        const int snap, int n_threads)
{

    if (n_threads < 0) {
        std::string error_message = ("ERROR number of threads cannot be"
                                     " negative");
        throw isce3::except::OutOfRange(ISCE_SRCINFO(), error_message);
    }
    if (min_block_size > max_block_size) {
        std::string error_message = ("ERROR minimum block size cannot be"
                                     "greater than the maximum block size");
        throw isce3::except::RuntimeError(ISCE_SRCINFO(), error_message);
    }

    // set initial values
    bool flag_2d = (block_width != nullptr || nblocks_x != nullptr);

    if (n_threads == 0) {
        n_threads = _omp_thread_count();
    }

    int min_block_length, max_block_length;
    int min_block_width = 1, max_block_width = 1;
    int _nblocks_y, _nblocks_x = 0;

    if (!flag_2d) {
        min_block_length = min_block_size / (static_cast<long long>(nbands) *
            array_width * type_size);
        max_block_length = max_block_size / (static_cast<long long>(nbands) *
            array_width * type_size);
        _nblocks_y = n_threads;
    } else {
        min_block_length = std::sqrt(min_block_size / (nbands * type_size));
        max_block_length = std::sqrt(max_block_size / (nbands * type_size));
        min_block_width = min_block_length;
        max_block_width = max_block_length;
        _nblocks_y = std::sqrt(n_threads);
        _nblocks_x = std::sqrt(n_threads);
    }

    // set nblocks and block size (Y-axis)
    _nblocks_y = std::max(_nblocks_y, 1);
    int _block_length =
            std::ceil((static_cast<float>(array_length)) / _nblocks_y);
    _block_length =
            std::max(_block_length, min_block_length);
    _block_length =
            std::min(_block_length, max_block_length);


    // set nblocks and block size (X-axis)
    int _block_width = array_width;
    if (flag_2d) {
        _nblocks_x = std::max(_nblocks_x, 1);
        _block_width =
                std::ceil((static_cast<float>(array_width)) / _nblocks_x);
        _block_width =
                std::max(_block_width, min_block_width);
        _block_width =
                std::min(_block_width, max_block_width);
    }

    // snap (_block_length multiple of snap) and update nblocks
    _block_length = std::round(_block_length / snap) * snap;
    _block_length = std::max(_block_length, 1);
    _block_length = std::min(_block_length, array_length);
    _nblocks_y = std::ceil(((float) array_length) / _block_length);

    if (flag_2d) {
        _block_width = std::round(_block_width / snap) * snap;
        _block_width = std::max(_block_width, 1);
        _block_width = std::min(_block_width, array_width);
        _nblocks_x = std::ceil(((float) array_width) / _block_width);
    }

    if (nblocks_x != nullptr)
        *nblocks_x = _nblocks_x;
    if (nblocks_y != nullptr)
        *nblocks_y = _nblocks_y;

    if (channel != nullptr) {
        *channel << "array length: " << array_length << pyre::journal::newline;
        *channel << "array width: " << array_width << pyre::journal::newline;
        *channel << "number of available thread(s): " << n_threads
                 << pyre::journal::newline;

        if (!flag_2d)
            *channel << "number of block(s): " << _nblocks_y
                     << pyre::journal::newline;
        else {
            *channel << "number of block(s): "
                     << " " << _nblocks_y << " x " << _nblocks_x
                     << " (Y x X) = " << _nblocks_y * _nblocks_x
                     << pyre::journal::newline;
        }
    }

    if (block_length != nullptr) {
        *block_length = _block_length;
        if (channel != nullptr) {
            *channel << "block length: " << *block_length
                     << pyre::journal::newline;
        }
    }

    if (block_width != nullptr) {
        *block_width = _block_width;
        if (channel != nullptr) {
            *channel << "block width: " << *block_width
                     << pyre::journal::newline;
        }
    }

    if (channel != nullptr) {

        long long block_size_bytes =
                ((static_cast<long long>(_block_length)) *
                 _block_width * nbands * type_size);

        std::string block_size_bytes_str = isce3::core::getNbytesStr(block_size_bytes);
        if (nbands > 1)
            block_size_bytes_str += " (" + std::to_string(nbands) + " bands)";

        *channel << "block size: " << block_size_bytes_str
                 << pyre::journal::endl;
    }
}

}}
