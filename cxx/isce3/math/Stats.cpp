#include "Stats.h"

#include <pyre/journal.h>
#include <isce3/math/complexOperations.h>


namespace isce3 {
namespace math {

// helpers

template<class T> T signedRealOrComplexModulus(T val) {
    return val;
}

template<class T> T signedRealOrComplexModulus(std::complex<T> val) {
    return std::abs(val);
}

template<class T>
bool isnan(T val) {
    return std::isnan(val);
}

template<class T>
bool isnan(std::complex<T> val) {
    return std::isnan(val.real()) || std::isnan(val.imag());
}

// Stats methods

template<class T>
void Stats<T>::update(const T & value)
{
    if (isnan(value)) {
        return;
    }
    n_valid += 1;

    // T mean (real or complex)
    mean += (value - mean) / static_cast<T_real>(n_valid);

    // Provisional mean
    const auto value_real = signedRealOrComplexModulus(value);
    const double delta = value_real - real_valued_mean;
    real_valued_mean += delta / n_valid;

    // Square diff sum
    square_diff_sum += delta * (value_real - real_valued_mean);

    // Max
    if (isnan(max) || value_real > signedRealOrComplexModulus(max)) {
        max = value;
    }

    // Min
    if (isnan(min) || value_real < signedRealOrComplexModulus(min)) {
        min = value;
    }
}


template<class T>
void Stats<T>::update(const Stats<T>& other) {
    if (other.n_valid <= 0) {
        return;
    }
    const auto nself = n_valid, ntotal = n_valid + other.n_valid;
    n_valid += other.n_valid;

    // T mean (real or complex)
    mean += (other.mean - mean) * (other.n_valid / static_cast<T_real>(ntotal));

    // Real-valued mean
    const double delta = other.real_valued_mean - real_valued_mean;
    real_valued_mean += delta * other.n_valid / ntotal;

    // Square diff sum
    square_diff_sum += other.square_diff_sum +
        std::pow(delta, 2) * nself * other.n_valid / ntotal;

    // Max
    if (isnan(max) ||
            signedRealOrComplexModulus(other.max) >
            signedRealOrComplexModulus(max)) {
        max = other.max;
    }

    // Min
    if (isnan(min) ||
            signedRealOrComplexModulus(other.min) <
            signedRealOrComplexModulus(min)) {
        min = other.min;
    }
}


template<class T>
Stats<T>::Stats(const T* values, size_t size, size_t stride)
{
    const T* end = values + size * stride;
    for (const T* ptr = values; ptr < end; ptr += stride) {
        update(*ptr);
    }
}


template<class T>
void Stats<T>::update(const T* values, size_t size, size_t stride)
{
    const Stats<T> block_stats(values, size, stride);
    update(block_stats);
}


template<class T>
typename Stats<T>::T_real Stats<T>::sample_stddev() const
{
    if (n_valid > 1) {
        return std::sqrt(square_diff_sum / (n_valid - 1));
    } else {
        return 0.0;
    }
}

// StatsRealImag methods

template<class T>
void StatsRealImag<T>::update(const std::complex<T>& value)
{
    // skip if either component is nan
    if (!isnan(value)) {
        real.update(value.real());
        imag.update(value.imag());
        n_valid = real.n_valid;
    }
}

template<class T>
void StatsRealImag<T>::update(const StatsRealImag<T>& other)
{
    real.update(other.real);
    imag.update(other.imag);
    n_valid = real.n_valid;
}


template<class T>
StatsRealImag<T>::StatsRealImag(const std::complex<T>* values,
        size_t size, size_t stride)
{
    using C = std::complex<T>;
    const C* end = values + size * stride;
    for (const C* ptr = values; ptr < end; ptr += stride) {
        update(*ptr);
    }
}


template<class T>
void StatsRealImag<T>::update(const std::complex<T>* values,
        size_t size, size_t stride)
{
    const StatsRealImag<T> block_stats(values, size, stride);
    update(block_stats);
}


// class template instantiations

template class Stats<float>;
template class Stats<double>;
template class Stats<std::complex<float>>;
template class Stats<std::complex<double>>;

template class StatsRealImag<float>;
template class StatsRealImag<double>;

// functions for parallel stats calculation over Raster objects.

template<class StatsT>
StatsT _aggregateStats(std::vector<StatsT>& statsvec)
{
    StatsT allstats;
    for (const auto stats : statsvec) {
        allstats.update(stats);
    }
    return allstats;
}


// StatsT could be Stats<T> or StatsRealImag<T>
template<class StatsT>
void _runBlock(isce3::io::Raster& input_raster,
               std::vector<std::vector<StatsT>>& block_stats_vector,
               const int block_count, const long x0, const long block_width,
               const long y0, const long block_length, const int band) {

    using T = typename StatsT::type;

    // Read band array from input_raster
    isce3::core::Matrix<T> block_array(block_length, block_width);

    _Pragma("omp critical")
    {
        input_raster.getBlock(block_array.data(), 
                              x0, y0, block_width,
                              block_length, band + 1);
    }

    // Get block stats pointer
    StatsT* block_stats = &block_stats_vector[band][block_count];

    block_stats->update(block_array.data(),
        block_array.length() * block_array.width());
}


template<class T>
std::vector<Stats<T>> computeRasterStats(
    isce3::io::Raster& input_raster,
    isce3::core::MemoryModeBlocksY memory_mode) {

    std::string info_str = "isce3.math.computeRasterStats";
    if (std::is_same<T, float>::value) {
        info_str += "<float>";
    }
    else if (std::is_same<T, double>::value) {
        info_str += "<double>";
    }
    else if (std::is_same<T, std::complex<float>>::value) {
        info_str += "<complex(float)>";
    }
    else if (std::is_same<T, std::complex<double>>::value) {
        info_str += "<complex(double)>";
    }

    pyre::journal::info_t info(info_str);

    const long x0 = 0;
    const int nbands = input_raster.numBands();
    info << "nbands: " << nbands << pyre::journal::endl;
    const long block_width = input_raster.width();
    int block_length, nblocks;


    if (memory_mode == isce3::core::MemoryModeBlocksY::SingleBlockY) {
        nblocks = 1;
        block_length = input_raster.length();
    } else {
        isce3::core::getBlockProcessingParametersY(
            input_raster.length(), input_raster.width(), 
            nbands, GDALGetDataTypeSizeBytes(input_raster.dtype()), 
            &info, &block_length, &nblocks);
    }

    std::vector<Stats<T>> stats_vector(nbands);
    std::vector<std::vector<Stats<T>>> block_stats_vector(
        nbands, std::vector<Stats<T>>(nblocks));

    for (int band = 0; band < input_raster.numBands(); ++band) {

        info << "processing band: " << band + 1 << pyre::journal::endl;

        _Pragma("omp parallel for")
        for (int block_count = 0; block_count < nblocks; ++block_count) {

            const long y0 = block_count * block_length;
            const long this_block_length = std::min(
                static_cast<long>(block_length), static_cast<long>(input_raster.length()) - y0);

            _runBlock(input_raster, block_stats_vector, block_count,
                      x0, block_width, y0, this_block_length, band);

        }
    }
        
    const auto n_elements = (static_cast<long long>(input_raster.width()) * 
                             input_raster.length());
    
    for (int band = 0; band < nbands; ++band) {

        stats_vector[band] = _aggregateStats(block_stats_vector[band]);

        info << "band: " << band + 1 << pyre::journal::newline
             << "    n. valid: " << stats_vector[band].n_valid 
             << " (" << 100 * stats_vector[band].n_valid / n_elements
             << "%) " << pyre::journal::newline
             << "    min: " << stats_vector[band].min
             << ", mean: " << stats_vector[band].mean
             << ", max: " << stats_vector[band].max
             << ", sample stddev: " << stats_vector[band].sample_stddev() 
             << pyre::journal::endl; 

    }
    return stats_vector;
}

template std::vector<Stats<float>>
    computeRasterStats(
        isce3::io::Raster& input_raster,
        isce3::core::MemoryModeBlocksY memory_mode);

template std::vector<Stats<double>>
    computeRasterStats(
        isce3::io::Raster& input_raster,
        isce3::core::MemoryModeBlocksY memory_mode);

template std::vector<Stats<std::complex<float>>>
    computeRasterStats(
        isce3::io::Raster& input_raster,
        isce3::core::MemoryModeBlocksY memory_mode);

template std::vector<Stats<std::complex<double>>>
    computeRasterStats(
        isce3::io::Raster& input_raster,
        isce3::core::MemoryModeBlocksY memory_mode);



template<class T>
std::vector<StatsRealImag<T>> computeRasterStatsRealImag(
    isce3::io::Raster& input_raster,
    isce3::core::MemoryModeBlocksY memory_mode) {

    pyre::journal::info_t info("isce3.math.computeRasterStatsRealImag");

    const long x0 = 0;
    const int nbands = input_raster.numBands();
    info << "nbands: " << nbands << pyre::journal::endl;
    const long block_width = input_raster.width();
    int block_length, nblocks;


    if (memory_mode == isce3::core::MemoryModeBlocksY::SingleBlockY) {
        nblocks = 1;
        block_length = input_raster.length();
    } else {
        isce3::core::getBlockProcessingParametersY(
            input_raster.length(), input_raster.width(), 
            nbands, GDALGetDataTypeSizeBytes(input_raster.dtype()), 
            &info, &block_length, &nblocks);
    }
    
    std::vector<StatsRealImag<T>> stats_vector(nbands);
    std::vector<std::vector<StatsRealImag<T>>> block_stats_vector(
        nbands, std::vector<StatsRealImag<T>>(nblocks));

    for (int band = 0; band < input_raster.numBands(); ++band) {

        info << "processing band: " << band + 1 << pyre::journal::endl;
        _Pragma("omp parallel for")
        for (int block_count = 0; block_count < nblocks; ++block_count) {

            const long y0 = block_count * block_length;
            const long this_block_length = std::min(
                (long) block_length, (long) input_raster.length() - y0);

            _runBlock(input_raster, block_stats_vector, block_count,
                      x0, block_width, y0, this_block_length, band);
        }
    }
        
    const auto n_elements = (static_cast<long long>(input_raster.width()) * 
                             input_raster.length());
    
    for (int band = 0; band < nbands; ++band) {

        stats_vector[band] = _aggregateStats(block_stats_vector[band]);

        info << "band: " << band + 1 << pyre::journal::newline
             << "    n. valid: " << stats_vector[band].n_valid 
             << " (" << 100 * stats_vector[band].n_valid / n_elements
             << "%) " << pyre::journal::newline

             << "    min (real): " << stats_vector[band].real.min
             << ", mean (real): " << stats_vector[band].real.mean
             << ", max (real): " << stats_vector[band].real.max
             << ", sample stddev (real): " << stats_vector[band].real.sample_stddev()
              << pyre::journal::newline

             << "    min (imag): " << stats_vector[band].imag.min
             << ", mean (imag): " << stats_vector[band].imag.mean
             << ", max (imag): " << stats_vector[band].imag.max
             << ", sample stddev (imag): " << stats_vector[band].imag.sample_stddev()
             << pyre::journal::endl; 

    }
    return stats_vector;
}

template std::vector<StatsRealImag<float>>
    computeRasterStatsRealImag(
        isce3::io::Raster& input_raster,
        isce3::core::MemoryModeBlocksY memory_mode);

template std::vector<StatsRealImag<double>>
    computeRasterStatsRealImag(
        isce3::io::Raster& input_raster,
        isce3::core::MemoryModeBlocksY memory_mode);


}}
