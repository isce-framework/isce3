#include <isce3/core/TypeTraits.h>
#include <isce3/core/Constants.h>
#include <isce3/io/Raster.h>

namespace isce3 { namespace math {

/** Statistics struct (real valued)
*/
template<class T>
struct Stats {
    using T_real = typename isce3::real<T>::type;
    T min = std::numeric_limits<T_real>::quiet_NaN();
    T max = std::numeric_limits<T_real>::quiet_NaN();
    T mean = 0;

    double sample_stddev = 0;
    long long n_valid = 0;
};

/** Statistics struct (complex valued)
 * 
 * For complex T, min and max are complex but they are 
 * selected using the elements' magnitudes. The sample 
 * standard deviation is real-valued calculated using
 * the elements' magnitudes.
 * 
*/
template<class T>
struct Stats<std::complex<T>> {
    std::complex<T> min = std::complex(std::numeric_limits<T>::quiet_NaN(),
                                       std::numeric_limits<T>::quiet_NaN());
    std::complex<T> max = std::complex(std::numeric_limits<T>::quiet_NaN(),
                                       std::numeric_limits<T>::quiet_NaN());
    std::complex<T> mean = 0;

    double sample_stddev = 0;
    long long n_valid = 0;
};

/** Statistics struct
 * 
 * Statistics are computed independently for real and imaginary
 * parts.
 * 
*/
template<class T>
struct StatsRealImag {
    using T_real = typename isce3::real<T>::type;
    T_real min_real = std::numeric_limits<T_real>::quiet_NaN();
    T_real max_real = std::numeric_limits<T_real>::quiet_NaN();
    T_real mean_real = 0;
    double sample_stddev_real = 0;

    T_real min_imag = std::numeric_limits<T_real>::quiet_NaN();
    T_real max_imag = std::numeric_limits<T_real>::quiet_NaN();
    T_real mean_imag = 0;
    double sample_stddev_imag = 0;

    long long n_valid = 0;
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
    isce3::core::MemoryModeBlockY memory_mode = 
        isce3::core::MemoryModeBlockY::AutoBlocksY);

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
    isce3::core::MemoryModeBlockY memory_mode = 
        isce3::core::MemoryModeBlockY::AutoBlocksY);


}}
