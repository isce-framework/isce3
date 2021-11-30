#include "Stats.h"

#include <pyre/journal.h>
#include <isce3/geocode/GeocodeCov.h>
#include <isce3/math/complexOperations.h>


namespace isce3 {
namespace math {

/** Block statistics struct */
template<class T>
struct BlockStats {
    using T_real = typename isce3::real<T>::type;

    T min = std::numeric_limits<T_real>::quiet_NaN();
    T max = std::numeric_limits<T_real>::quiet_NaN();
    T mean = 0;

    // Real variables to compute the stddev
    double real_valued_mean = 0;
    double square_diff_sum = 0;
    long long n_valid = 0;
};

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


template<class T>
inline void _updateStats(BlockStats<T> * block_stats, T & geo_value) {
                
    /* Compute stats with the Welford's online algorithm */

    using T_real = typename isce3::real<T>::type;

    // n_valid
    block_stats->n_valid += 1;

    // T mean (real or complex)
    block_stats->mean += ((geo_value - block_stats->mean) / 
                           static_cast<T_real>(block_stats->n_valid));

    // Provisional mean
    auto geo_value_real = signedRealOrComplexModulus(geo_value);
    double delta = (geo_value_real - 
                    block_stats->real_valued_mean);
    block_stats->real_valued_mean += (delta / block_stats->n_valid);

    // Square diff sum
    block_stats->square_diff_sum += (delta *
        (geo_value_real - block_stats->real_valued_mean));

    // Max
    if (isnan(block_stats->max) || geo_value_real >
            signedRealOrComplexModulus(block_stats->max)) {
        block_stats->max = geo_value;
    }

    // Min
    if (isnan(block_stats->min) || geo_value_real <
            signedRealOrComplexModulus(block_stats->min)) {
        block_stats->min = geo_value;
    }
}


template<class T>
void _writeStatsToStruct(
        Stats<T>& stats,
        T& min, T& max, T& mean, double stddev, long long n_valid)
{
    stats.min = min; 
    stats.max = max; 
    stats.mean = mean; 
    stats.sample_stddev = stddev; 
    stats.n_valid = n_valid;
}

template<class T>
void _saveStats(Stats<T>& stats,
                std::vector<BlockStats<T>>& block_stats_vector_band) {

    using T_real = typename isce3::real<T>::type;

    // Compute stats
    T min = std::numeric_limits<T_real>::quiet_NaN();
    T max = std::numeric_limits<T_real>::quiet_NaN();
    T mean = 0;

    double real_valued_mean = 0;
    double square_diff_sum = 0;
    long long n_valid = 0;

    for (auto block_stats: block_stats_vector_band) {

        // Real- or complex-valued max
        if (isnan(max) || 
                signedRealOrComplexModulus(block_stats.max) > 
                signedRealOrComplexModulus(max)) {
            max = block_stats.max;
        }

        // Real- or complex-valued min
        if (isnan(min) || 
                signedRealOrComplexModulus(block_stats.min) < 
                signedRealOrComplexModulus(min)) {
            min = block_stats.min;
        }
    
        /* Compute stats between sets with the Chan's method */
        
        // N. valid pixels
        const double n_a = n_valid;
        const double n_b = block_stats.n_valid;
        n_valid += block_stats.n_valid;
        const double n_ab = n_valid;

        // T-valued mean
        mean += (block_stats.mean - mean) * static_cast<T_real> (n_b / n_ab);

        // Real-valued mean
        double delta = (block_stats.real_valued_mean - real_valued_mean);
        real_valued_mean += delta * n_b / n_ab;

        // Square diff sum
        square_diff_sum += (block_stats.square_diff_sum + 
                            std::pow(delta, 2) * n_a * n_b / n_ab);

    }

    const double stddev = std::sqrt(square_diff_sum / ((double) n_valid - 1));

    _writeStatsToStruct(stats, min, max, mean, stddev, n_valid);
}

template<class T>
void _runBlock(isce3::io::Raster& input_raster,
               std::vector<std::vector<BlockStats<T>>>& block_stats_vector,
               const int block_count, const long x0, const long block_width,
               const long y0, const long block_length, const int band) {

    // Read band array from input_raster
    isce3::core::Matrix<T> block_array(block_length, block_width);

    _Pragma("omp critical")
    {
        input_raster.getBlock(block_array.data(), 
                              x0, y0, block_width,
                              block_length, band + 1);
    }

    // Get block stats pointer
    isce3::math::BlockStats<T> * block_stats =
        & block_stats_vector[band][block_count];

    for (long i = 0; i < block_length; ++i) {
        for (long j = 0; j < block_width; ++j) {
            auto val = block_array(i, j);
            if (isnan(val)) {
                continue;
            }
            // Update block stats
            _updateStats(block_stats, val);

        }
    }
}


template<class T_real>
void _runBlockRealImag(isce3::io::Raster& input_raster,
                       std::vector<std::vector<BlockStats<T_real>>>& block_stats_vector_real,
                       std::vector<std::vector<BlockStats<T_real>>>& block_stats_vector_imag,
                       const int block_count, const long x0, const long block_width,
                       const long y0, const long block_length, const int band) {

    // Read band array from input_raster
    isce3::core::Matrix<std::complex<T_real>> block_array(block_length, block_width);

    _Pragma("omp critical")
    {
        input_raster.getBlock(block_array.data(), 
                              x0, y0, block_width,
                              block_length, band + 1);
    }

    // Get block stats pointers
    isce3::math::BlockStats<T_real> * block_stats_real =
        & block_stats_vector_real[band][block_count];
    isce3::math::BlockStats<T_real> * block_stats_imag =
        & block_stats_vector_imag[band][block_count];

    for (long i = 0; i < block_length; ++i) {
        for (long j = 0; j < block_width; ++j) {
            auto val = block_array(i, j);
            if (isnan(val)) {
                continue;
            }
            // Update block stats
            T_real val_real = val.real(), val_imag = val.imag();
            _updateStats(block_stats_real, val_real);
            _updateStats(block_stats_imag, val_imag);

        }
    }
}


template<class T>
std::vector<isce3::math::Stats<T>> computeRasterStats(
    isce3::io::Raster& input_raster,
    isce3::core::MemoryModeBlockY memory_mode) {

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


    if (memory_mode == isce3::core::MemoryModeBlockY::SingleBlockY) {
        nblocks = 1;
        block_length = input_raster.length();
    } else {
        isce3::geocode::getBlocksNumberAndLength(
            input_raster.length(), input_raster.width(), 
            nbands, GDALGetDataTypeSizeBytes(input_raster.dtype()), 
            &info, &block_length, &nblocks);
    }

    std::vector<isce3::math::Stats<T>> stats_vector(nbands);
    std::vector<std::vector<isce3::math::BlockStats<T>>> block_stats_vector(
        nbands, std::vector<isce3::math::BlockStats<T>>(nblocks));

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

        _saveStats(stats_vector[band], block_stats_vector[band]);

        info << "band: " << band + 1 << pyre::journal::newline
             << "    n. valid: " << stats_vector[band].n_valid 
             << " (" << 100 * stats_vector[band].n_valid / n_elements
             << "%) " << pyre::journal::newline
             << "    min: " << stats_vector[band].min
             << ", mean: " << stats_vector[band].mean
             << ", max: " << stats_vector[band].max
             << ", sample stddev: " << stats_vector[band].sample_stddev 
             << pyre::journal::endl; 

    }
    return stats_vector;
}

template std::vector<isce3::math::Stats<float>>
    computeRasterStats(
        isce3::io::Raster& input_raster,
        isce3::core::MemoryModeBlockY memory_mode);

template std::vector<isce3::math::Stats<double>>
    computeRasterStats(
        isce3::io::Raster& input_raster,
        isce3::core::MemoryModeBlockY memory_mode);

template std::vector<isce3::math::Stats<std::complex<float>>>
    computeRasterStats(
        isce3::io::Raster& input_raster,
        isce3::core::MemoryModeBlockY memory_mode);

template std::vector<isce3::math::Stats<std::complex<double>>>
    computeRasterStats(
        isce3::io::Raster& input_raster,
        isce3::core::MemoryModeBlockY memory_mode);



template<class T>
std::vector<isce3::math::StatsRealImag<T>> computeRasterStatsRealImag(
    isce3::io::Raster& input_raster,
    isce3::core::MemoryModeBlockY memory_mode) {

    pyre::journal::info_t info("isce3.math.computeRasterStatsRealImag");

    const long x0 = 0;
    const int nbands = input_raster.numBands();
    info << "nbands: " << nbands << pyre::journal::endl;
    const long block_width = input_raster.width();
    int block_length, nblocks;


    if (memory_mode == isce3::core::MemoryModeBlockY::SingleBlockY) {
        nblocks = 1;
        block_length = input_raster.length();
    } else {
        isce3::geocode::getBlocksNumberAndLength(
            input_raster.length(), input_raster.width(), 
            nbands, GDALGetDataTypeSizeBytes(input_raster.dtype()), 
            &info, &block_length, &nblocks);
    }
    
    std::vector<isce3::math::StatsRealImag<T>> stats_vector(nbands);
    std::vector<isce3::math::Stats<T>> stats_vector_real(nbands);
    std::vector<isce3::math::Stats<T>> stats_vector_imag(nbands);
    std::vector<std::vector<isce3::math::BlockStats<T>>> block_stats_vector_real(
        nbands, std::vector<isce3::math::BlockStats<T>>(nblocks));
    std::vector<std::vector<isce3::math::BlockStats<T>>> block_stats_vector_imag(
        nbands, std::vector<isce3::math::BlockStats<T>>(nblocks));

    for (int band = 0; band < input_raster.numBands(); ++band) {

        info << "processing band: " << band + 1 << pyre::journal::endl;
        _Pragma("omp parallel for")
        for (int block_count = 0; block_count < nblocks; ++block_count) {

            const long y0 = block_count * block_length;
            const long this_block_length = std::min(
                (long) block_length, (long) input_raster.length() - y0);

            _runBlockRealImag(input_raster, block_stats_vector_real,
                              block_stats_vector_imag, block_count,
                              x0, block_width, y0, this_block_length, band);

        }
    }
        
    const auto n_elements = (static_cast<long long>(input_raster.width()) * 
                             input_raster.length());
    
    for (int band = 0; band < nbands; ++band) {

        _saveStats(stats_vector_real[band], block_stats_vector_real[band]);
        _saveStats(stats_vector_imag[band], block_stats_vector_imag[band]);

        stats_vector[band].min_real = stats_vector_real[band].min;
        stats_vector[band].max_real = stats_vector_real[band].max;
        stats_vector[band].mean_real = stats_vector_real[band].mean;
        stats_vector[band].sample_stddev_real = stats_vector_real[band].sample_stddev;
        
        stats_vector[band].min_imag = stats_vector_imag[band].min;
        stats_vector[band].max_imag = stats_vector_imag[band].max;
        stats_vector[band].mean_imag = stats_vector_imag[band].mean;
        stats_vector[band].sample_stddev_imag = stats_vector_imag[band].sample_stddev;

        stats_vector[band].n_valid = stats_vector_real[band].n_valid;

        info << "band: " << band + 1 << pyre::journal::newline
             << "    n. valid: " << stats_vector[band].n_valid 
             << " (" << 100 * stats_vector[band].n_valid / n_elements
             << "%) " << pyre::journal::newline

             << "    min (real): " << stats_vector[band].min_real
             << ", mean (real): " << stats_vector[band].mean_real
             << ", max (real): " << stats_vector[band].max_real
             << ", sample stddev (real): " << stats_vector[band].sample_stddev_real
              << pyre::journal::newline

             << "    min (imag): " << stats_vector[band].min_imag
             << ", mean (imag): " << stats_vector[band].mean_imag
             << ", max (imag): " << stats_vector[band].max_imag
             << ", sample stddev (imag): " << stats_vector[band].sample_stddev_imag
             << pyre::journal::endl; 

    }
    return stats_vector;
}

template std::vector<isce3::math::StatsRealImag<float>>
    computeRasterStatsRealImag(
        isce3::io::Raster& input_raster,
        isce3::core::MemoryModeBlockY memory_mode);

template std::vector<isce3::math::StatsRealImag<double>>
    computeRasterStatsRealImag(
        isce3::io::Raster& input_raster,
        isce3::core::MemoryModeBlockY memory_mode);

  
}}
