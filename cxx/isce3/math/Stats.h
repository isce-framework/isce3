#include <isce3/core/TypeTraits.h>
#include <isce3/core/Constants.h>
#include <isce3/core/blockProcessing.h>
#include <isce3/io/Raster.h>

namespace isce3 { namespace math {

/** Statistics struct
 *
 * For complex T, min and max are complex but they are
 * selected using the elements' magnitudes. The sample
 * standard deviation is real-valued calculated using
 * the elements' magnitudes.
*/
template<class T>
struct Stats {
    /** Expected element type of input data. */
    using type = T;
    using T_real = typename isce3::real<T>::type;
    T min = std::numeric_limits<T_real>::quiet_NaN();
    T max = std::numeric_limits<T_real>::quiet_NaN();
    T mean = 0;

    long long n_valid = 0;

    T_real sample_stddev() const;

    /** Update statistics with independent data using Chan's method */
    void update(const Stats<T>& other);

    /** Accumulate a data point using Welford's online algorithm. */
    void update(const T& value);

    /** Calculate stats of a new block of data using Welford's algorithm and
     *  update current estimate with Chan's method. */
    void update(const T* values, size_t size, size_t stride = 1);

    /** Initialize stats from block of data. */
    Stats(const T* values, size_t size, size_t stride = 1);

    Stats() = default;

private:
    double real_valued_mean = 0;
    double square_diff_sum = 0;
};

/** Statistics struct
 * 
 * Statistics are computed independently for real and imaginary
 * parts.
 * 
*/
template<class T>
struct StatsRealImag {
    /** Expected element type of input data. */
    using type = std::complex<T>;

    Stats<T> real;
    Stats<T> imag;
    long long n_valid = 0;

    /** Update statistics with independent data using Chan's method */
    void update(const StatsRealImag<T>& other);

    /** Accumulate a data point using Welford's online algorithm. */
    void update(const std::complex<T>& value);

    /** Calculate stats of a new block of data using Welford's algorithm and
     *  update current estimate with Chan's method. */
    void update(const std::complex<T>* values, size_t size, size_t stride = 1);

    /** Initialize from block of data. */
    StatsRealImag(const std::complex<T>* values, size_t size, size_t stride = 1);

    StatsRealImag() = default;
};

/** Compute raster statistics.
 *
 * Calculate statistics (min, max, mean, and standard deviation) 
 * from a multi-band raster.
 *
 * @param[in]  input_raster  Input raster
 * @param[out] memory_mode   Memory mode
 * @returns                  Returns stats (Stats) vector
 */
template<class T>
std::vector<isce3::math::Stats<T>> computeRasterStats(
    isce3::io::Raster& input_raster,
    isce3::core::MemoryModeBlocksY memory_mode = 
        isce3::core::MemoryModeBlocksY::AutoBlocksY);

/** Compute real and imaginary statistics separately from a complex-valued
 * raster.
 *
 * Calculate real and imaginary statistics from a multi-band raster.
 *
 * @param[in]  input_raster  Input raster
 * @param[out] memory_mode   Memory mode
 * @returns                  Returns stats (StatsRealImag) vector
 */
template<class T>
std::vector<isce3::math::StatsRealImag<T>> computeRasterStatsRealImag(
    isce3::io::Raster& input_raster, 
    isce3::core::MemoryModeBlocksY memory_mode = 
        isce3::core::MemoryModeBlocksY::AutoBlocksY);

}}
