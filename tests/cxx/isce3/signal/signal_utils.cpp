#include <cmath>
#include <complex>
#include <gtest/gtest.h>

#include <isce3/core/Utilities.h>

#include <isce3/signal/signalUtils.h>

TEST(SignalUtilsTest, UpsampleRasterBlock)
{

    size_t width = 100;
    size_t length = 1;
    size_t upsample_factor = 2;

   // create input complex array; 
   isce3::core::Matrix<std::complex<double>> slc(length, width);

   // a simple band limited signal (a linear phase ramp)
   for (size_t j = 0; j < width; ++j) {
       double phase = 2 * M_PI * j * 0.01;
       for (size_t i = 0; i < length; ++i) {
           slc(i, j) = std::complex<double>(std::cos(phase), std::sin(phase));
       }
    }

    // create raster and copy values
    std::string vsimem_ref = "/vsimem/" + getTempString("signal_utils_unitest");
    isce3::io::Raster input_raster(vsimem_ref, width,
                                   length, 1, GDT_CFloat64, "ENVI");
    input_raster.setBlock(slc.data(), 0, 0, width, length, 1);

    // create output complex array;
    std::valarray<std::complex<double>> output_array(length * width *
                                                    upsample_factor);

    // run upsampleRasterBlockX();
    isce3::signal::upsampleRasterBlockX(input_raster, output_array, 0, 0, width,
                                        length, 1, upsample_factor);
                                       
   // max error tolerance
   double max_arg_err = 0.0;
   double accumulated_arg_err = 0;
   int margin = 4;
   std::valarray<std::complex<double>> slcU(width * upsample_factor);
   std::valarray<std::complex<double>> expSlcU(width * upsample_factor);
 
   for (size_t j = margin ; j < width * upsample_factor - margin; ++j) {
       double phase = 2 * M_PI * j * 0.01 / upsample_factor;
       const std::complex<double> expected_signal =
               std::complex<double>(std::cos(phase), std::sin(phase));
       for (size_t i = 0; i < length; ++i) {
           std::complex<double> diff =
                   (output_array[i * width + j] * std::conj(expected_signal));
            accumulated_arg_err += std::arg(diff);
           if (std::abs(std::arg(diff)) > max_arg_err)
               max_arg_err = std::abs(std::arg(diff));
       }
   }

   double mean_arg_err = accumulated_arg_err / (width * upsample_factor - 2 * margin);
   std::cout << "maximum phase error : " << max_arg_err << std::endl;
   std::cout << "mean phase error : " << mean_arg_err << std::endl;
   ASSERT_LT(max_arg_err, 1.0e-15);
   ASSERT_LT(std::abs(mean_arg_err), 1.0e-16); 

}

int main(int argc, char * argv[]) {
      testing::InitGoogleTest(&argc, argv);
      return RUN_ALL_TESTS();
}
