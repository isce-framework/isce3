
#include <iostream>
#include <cstdio>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <valarray>
#include <complex>
#include <gtest/gtest.h>

#include "isce/signal/Signal.h"
#include "isce/io/Raster.h"

TEST(Signal, ForwardBackwardRangeFloat)
{
    // take a block of data, perform range FFT and then iverse FFT and compare with original data   
    isce::io::Raster inputSlc("../data/warped_envisat.slc.vrt");

    int width = inputSlc.width();
    int length = inputSlc.length();

    // 
    int blockLength = length;

    // reserve memory for a block of data
    std::valarray<std::complex<float>> data(width*blockLength);

    // reserve memory for the spectrum of the block of data
    std::valarray<std::complex<float>> range_spectrum(width*blockLength);

    // reserve memory for a block of data computed from inverse FFT
    std::valarray<std::complex<float>> invertData(width*blockLength);

    // a signal object
    isce::signal::Signal<float> sig;

    // create the forward and backward plans
    sig.forwardRangeFFT(data, range_spectrum, width, blockLength);
    sig.inverseRangeFFT(range_spectrum, invertData, width, blockLength);

    // read a block of data
    inputSlc.getBlock(data, 0, 0, width, blockLength);
   
    // forward fft transform
    sig.forward(data, range_spectrum);

    // inverse fft transform
    sig.inverse(range_spectrum, invertData);

    //normalize the result of inverse fft 
    invertData /=width;

    int blockSize = width*blockLength;
    std::complex<float> err(0.0, 0.0);
    double max_err = 0.0;
    for ( size_t i = 0; i < blockSize; ++i ) {
        err = invertData[i] - data[i];
        if (std::abs(err) > max_err){
            max_err = std::abs(err);
        }
    }

    ASSERT_LT(max_err, 1.0e-4);
}

TEST(Signal, ForwardBackwardAzimuthFloat)
{
      // take a block of data, perform azimuth FFT and then iverse FFT and compare with original data
      isce::io::Raster inputSlc("../data/warped_envisat.slc.vrt");

      int width = inputSlc.width();
      int length = inputSlc.length();

      //
      int blockLength = length;

      // reserve memory for a block of data
      std::valarray<std::complex<float>> data(width*blockLength);

      // reserve memory for the spectrum of the block of data
      std::valarray<std::complex<float>> azimuth_spectrum(width*blockLength);

      // reserve memory for a block of data computed from inverse FFT
      std::valarray<std::complex<float>> invertData(width*blockLength);

      // a signal object
      isce::signal::Signal<float> sig;

      // create the forward and backward plans
      sig.forwardAzimuthFFT(data, azimuth_spectrum, width, blockLength);
      sig.inverseAzimuthFFT(azimuth_spectrum, invertData, width, blockLength);

      // read a block of data
      inputSlc.getBlock(data, 0, 0, width, blockLength);

      // forward fft transform
      sig.forward(data, azimuth_spectrum);

      // inverse fft transform
      sig.inverse(azimuth_spectrum, invertData);

      //normalize the result of inverse fft
      invertData /=length;

      int blockSize = width*blockLength;
      std::complex<float> err(0.0, 0.0);
      double max_err = 0.0;
      for ( size_t i = 0; i < blockSize; ++i ) {
          err = invertData[i] - data[i];
          if (std::abs(err) > max_err){
              max_err = std::abs(err);
          }
      }
      ASSERT_LT(max_err, 1.0e-4);
}


TEST(Signal, nfft)
{
    // This test is same as the previous test but with nfft used for FFT computation instead of number of columns
    isce::io::Raster inputSlc("../data/warped_envisat.slc.vrt");

    int width = inputSlc.width();
    int length = inputSlc.length();
    //width = 51;
    int blockLength = length;
    
    // fft length for FFT computations 
    int nfft = 512;



    // reserve memory for a block of data with the size of nfft
    std::valarray<std::complex<float>> data(nfft*blockLength);

    // reserve memory for the spectrum of the block of data
    std::valarray<std::complex<float>> range_spectrum(nfft*blockLength);

    // reserve memory for the block of data after inverse fft
    std::valarray<std::complex<float>> invertData(nfft*blockLength);

    data = 0;
    range_spectrum = 0;
    invertData = 0;

    // instantiate a signal object
    isce::signal::Signal<float> sig;

    // create the forward and backward plans
    sig.forwardRangeFFT(data, range_spectrum, nfft, blockLength);
    sig.inverseRangeFFT(range_spectrum, invertData, nfft, blockLength);


    // read a block of data
    std::valarray<std::complex<float>> dataLine(width);
    for (size_t line = 0; line<blockLength; ++line){
        inputSlc.getLine(dataLine, line);
        data[std::slice(line*nfft,width,1)] = dataLine; //[std::slice(0,width,0)];
    }
    //inputSlc.getBlock(data, 0, 0, width, blockLength);

    // forward fft transform
    sig.forward(data, range_spectrum);

    // inverse fft transform
    sig.inverse(range_spectrum, invertData);

    //normalize the result of inverse fft
    invertData /=nfft;

    /*isce::io::Raster outputSlc("secSlc.slc", nfft, length, 1, GDT_CFloat32, "ENVI");
    outputSlc.setBlock(invertData, 0 ,0 , nfft, length);
    */ 

    std::complex<float> err(0.0, 0.0);
    double max_err = 0.0;
    
    for (size_t line = 0; line<blockLength; line++){
        for (size_t col = 0; col<width; col++){
            err = invertData[line*nfft+col] - data[line*nfft+col];
            if (std::abs(err) > max_err){
                    max_err = std::abs(err);
            }
        }
    }

    ASSERT_LT(max_err, 1.0e-4);   

}



TEST(Signal, nfftDouble)
{
    int width = 100; //16;
    int length = 1;
    int blockLength = length;

    // fft length for FFT computations
    size_t nfft;

    // instantiate a signal object
    isce::signal::Signal<double> sig;
    sig.nextPowerOfTwo(width, nfft);
    //nfft = width;

    // reserve memory for a block of data with the size of nfft
    std::valarray<std::complex<double>> data(nfft);
    std::valarray<std::complex<double>> spec(nfft);
    std::valarray<std::complex<double>> dataInv(nfft);

    for (size_t i=0; i<width; ++i){
        double phase = std::sin(10*M_PI*i/width);
        data[i] = std::complex<double> (std::cos(phase), std::sin(phase));
    }

    sig.forwardRangeFFT(data, spec, nfft, 1);
    sig.inverseRangeFFT(spec, dataInv, nfft, 1);
    
    sig.forward(data, spec);
    sig.inverse(spec, dataInv);

    dataInv /=nfft; 

    std::complex<double> err(0.0, 0.0);
    double max_err = 0.0;

    for (size_t line = 0; line<blockLength; line++){
          for (size_t col = 0; col<width; col++){
              err = dataInv[line*nfft+col] - data[line*nfft+col];
              if (std::abs(err) > max_err){
                      max_err = std::abs(err);
              }
          }
     }

    ASSERT_LT(max_err, 1.0e-12);
}


TEST(Signal, nfftFloat)
{
      int width = 100; //16;
      int length = 1;
      int blockLength = length;

      // fft length for FFT computations
      size_t nfft;

      // instantiate a signal object
      isce::signal::Signal<float> sig;
      sig.nextPowerOfTwo(width, nfft);
      //nfft = width;

      // reserve memory for a block of data with the size of nfft
      std::valarray<std::complex<float>> data(nfft);
      std::valarray<std::complex<float>> spec(nfft);
      std::valarray<std::complex<float>> dataInv(nfft);

      for (size_t i=0; i<width; ++i){
          double phase = std::sin(10*M_PI*i/width);
          data[i] = std::complex<float> (std::cos(phase), std::sin(phase));
      }

      sig.forwardRangeFFT(data, spec, nfft, 1);
      sig.inverseRangeFFT(spec, dataInv, nfft, 1);

      sig.forward(data, spec);
      sig.inverse(spec, dataInv);

      dataInv /=nfft;

      std::complex<float> err(0.0, 0.0);
      double max_err = 0.0;

      for (size_t line = 0; line<blockLength; line++){
            for (size_t col = 0; col<width; col++){
                err = dataInv[line*nfft+col] - data[line*nfft+col];
                if (std::abs(err) > max_err){
                        max_err = std::abs(err);
                }
            }
       }


      // average L2 error as described on 
      // http://www.fftw.org/accuracy/method.html
      std::complex<float> errorL2 ;
      std::complex<float> sumErr;
      std::complex<float> sumB;
      for (size_t line = 0; line<blockLength; line++){
              for (size_t col = 0; col<width; col++){
                  sumErr += std::pow((dataInv[line*nfft+col] - data[line*nfft+col]),2);
                  sumB += std::pow(dataInv[line*nfft+col], 2);
              }
         }
 
      errorL2 = std::sqrt(sumErr)/std::sqrt(sumB);
      
      ASSERT_LT(max_err, 1.0e-4);
      ASSERT_LT(std::abs(errorL2), 1.0e-4);
}

TEST(Signal, upsample)
{
    int width = 100; //16;

    // fft length for FFT computations
    size_t nfft;

    //sig.nextPowerOfTwo(width, nfft);
    nfft = width;
    // upsampling factor
    int oversample = 2;

    // reserve memory for a block of data with the size of nfft
    std::valarray<std::complex<double>> slc(nfft);
    std::valarray<std::complex<double>> spec(nfft);
    std::valarray<std::complex<double>> slcU(nfft*oversample);
    std::valarray<std::complex<double>> specU(nfft*oversample);

    for (size_t i=0; i<width; ++i){
        double phase = std::sin(10*M_PI*i/width);
        slc[i] = std::complex<double> (std::cos(phase), std::sin(phase));
    }


    // instantiate a signal object
    isce::signal::Signal<double> sig;
 
    sig.forwardRangeFFT(slc, spec, nfft, 1);
    sig.inverseRangeFFT(specU, slcU, nfft*oversample, 1);
    sig.upsample(slc, slcU, 1, nfft, oversample);

    // Check if the original smaples have the same phase in the signal before and after upsampling
    double max_err = 0.0;
    double err;
    for (size_t col = 0; col<width; col++){
        err = std::arg(slc[col] * std::conj(slcU[oversample*col]));
        if (std::abs(err) > max_err){
            max_err = std::abs(err);
        }
    }

    double max_err_u = 0.0;
    double err_u;
    double step = 1.0/oversample;
    std::complex<double> cpxData;
    for (size_t col = 0; col<width*oversample; col++){
        double i = col*step;
        double phase = std::sin(10*M_PI*i/(width));
        cpxData = std::complex<double> (std::cos(phase), std::sin(phase));
        err_u = std::arg(cpxData * std::conj(slcU[col]));
        if (std::abs(err_u) > max_err_u){
              max_err_u = std::abs(err_u);
        }
    }    

    std::cout << "max_err " << max_err << std::endl;
    std::cout << "max_err_u " << max_err_u << std::endl;
    ASSERT_LT(max_err, 1.0e-14);
    ASSERT_LT(max_err_u, 1.0e-9);

}

TEST(Signal, FFT2D)
  {

        int width = 12;
        int length = 10;

        //
        int blockLength = length;

        // reserve memory for a block of data
        std::valarray<std::complex<double>> data(width*blockLength);

        // reserve memory for the spectrum of the block of data
        std::valarray<std::complex<double>> spectrum(width*blockLength);

        // reserve memory for a block of data computed from inverse FFT
        std::valarray<std::complex<double>> invertData(width*blockLength);

        // a signal object
        isce::signal::Signal<double> sig;

        // create the forward and backward plans
        sig.forward2DFFT(data, spectrum, width, blockLength);
        sig.inverse2DFFT(spectrum, invertData, width, blockLength);

        for (size_t i = 0; i< length; ++i){
            for (size_t j = 0; j< width; ++j){
                data[i*width + j] = std::complex<double> (std::cos(i*j), std::sin(i*j));
            }
        }
  
        sig.forward(data, spectrum);
        sig.inverse(spectrum, invertData);
        invertData /=(width*length);

        double max_err = 0.0;
        double err = 0.0;
        for (size_t i = 0; i< length; ++i){
            for (size_t j = 0; j< width; ++j){
                err = std::abs(data[i*width + j] - invertData[i*width + j]);
                if (err > max_err)
                    max_err = err;
            }
        }

        ASSERT_LT(max_err, 1.0e-12);
}

TEST(Signal, MultiThread)
{

    int width = 1000;
    int length = 1000;

    // length of a block
    int blockLength = length;

    // reserve memory for a block of data
    std::valarray<std::complex<double>> data(width*blockLength);

    // reserve memory for the spectrum of the block of data
    std::valarray<std::complex<double>> spectrum(width*blockLength);

    // reserve memory for a block of data computed from inverse FFT
    std::valarray<std::complex<double>> invertData(width*blockLength);

    // a signal object
    isce::signal::Signal<double> sig(4);

    // create the forward and backward plans
    sig.forwardRangeFFT(data, spectrum, width, blockLength);
    sig.inverseRangeFFT(spectrum, invertData, width, blockLength);

    for (size_t i = 0; i< length; ++i){
        for (size_t j = 0; j< width; ++j){
            data[i*width + j] = std::complex<double> (std::cos(i*j), std::sin(i*j));
        }
    }

    sig.forward(data, spectrum);
    sig.inverse(spectrum, invertData);
    invertData /=(width);

    double max_err = 0.0;
    double err = 0.0;
    for (size_t i = 0; i< length; ++i){
        for (size_t j = 0; j< width; ++j){
            err = std::abs(data[i*width + j] - invertData[i*width + j]);
            if (err > max_err)
                max_err = err;
              }
    }

    ASSERT_LT(max_err, 1.0e-12);
}

TEST(Signal, rawPointerArrayComplex)
{
    int width = 120;
    int length = 100;

    int blockLength = length;

    // reserve memory for a block of data
    std::complex<double> *data = new std::complex<double>[width*blockLength];
    
    // reserve memory for the spectrum of the block of data
    std::complex<double> *spectrum = new std::complex<double>[width*blockLength];
    
    // reserve memory for a block of data computed from inverse FFT
    std::complex<double> *invertData = new std::complex<double>[width*blockLength];
    
    // a signal object
    isce::signal::Signal<double> sig;

    for (size_t i = 0; i< length; ++i){
        for (size_t j = 0; j< width; ++j){
            data[i*width + j] = std::complex<double> (std::cos(i*j), std::sin(i*j));
        }
    }

    // ********************************
    // create the forward and backward plans
    sig.forward2DFFT(data, spectrum, width, blockLength);
    sig.inverse2DFFT(spectrum, invertData, width, blockLength);

    // forward and inverse transforms
    sig.forward(data, spectrum);
    sig.inverse(spectrum, invertData);

    for (size_t i = 0; i< length; ++i){
        for (size_t j = 0; j< width; ++j){
            invertData[i*width + j] /=(width*length);
        }
    }

    double max_err_2DFFT = 0.0;
    double err = 0.0;
    for (size_t i = 0; i< length; ++i){
        for (size_t j = 0; j< width; ++j){
            err = std::abs(data[i*width + j] - invertData[i*width + j]);
            if (err > max_err_2DFFT)
                max_err_2DFFT = err;
        }
    }

    // ********************************
    // create the forward and backward plans
      sig.forwardRangeFFT(data, spectrum, width, blockLength);
      sig.inverseRangeFFT(spectrum, invertData, width, blockLength);

      // forward and inverse transforms
      sig.forward(data, spectrum);
      sig.inverse(spectrum, invertData);

      for (size_t i = 0; i< length; ++i){
          for (size_t j = 0; j< width; ++j){
              invertData[i*width + j] /=(width);
          }
      }

      double max_err_range = 0.0;
      err = 0.0;
      for (size_t i = 0; i< length; ++i){
          for (size_t j = 0; j< width; ++j){
              err = std::abs(data[i*width + j] - invertData[i*width + j]);
              if (err > max_err_range)
                  max_err_range = err;
          }
      }
      
    // ********************************
    // create the forward and backward plans
    sig.forwardAzimuthFFT(data, spectrum, width, blockLength);
    sig.inverseAzimuthFFT(spectrum, invertData, width, blockLength);

    // forward and inverse transforms
    sig.forward(data, spectrum);
    sig.inverse(spectrum, invertData);

    for (size_t i = 0; i< length; ++i){
        for (size_t j = 0; j< width; ++j){
            invertData[i*width + j] /=(length);
        }
    }

    double max_err_az = 0.0;
    err = 0.0;
    for (size_t i = 0; i< length; ++i){
        for (size_t j = 0; j< width; ++j){
            err = std::abs(data[i*width + j] - invertData[i*width + j]);
            if (err > max_err_az)
                max_err_az = err;
        }
    }

    // ********************************
    // Check the error for 2D FFT
    ASSERT_LT(max_err_2DFFT, 1.0e-12);

    // Check the error for FFT in range
    ASSERT_LT(max_err_range, 1.0e-12);

    // Check the error for FFT in azimuth
    ASSERT_LT(max_err_az, 1.0e-12);
}

TEST(Signal, realDataFFT)
{
      int width = 120;
      int length = 100;

      int blockLength = length;

      // reserve memory for a block of data
      double *data = new double[width*blockLength];

      // reserve memory for the spectrum of the block of data
      std::complex<double> *spectrum = new std::complex<double>[width*blockLength];

      // reserve memory for a block of data computed from inverse FFT
      double *invertData = new double[width*blockLength];

      // a signal object
      isce::signal::Signal<double> sig;

      for (size_t i = 0; i< length; ++i){
          for (size_t j = 0; j< width; ++j){
              data[i*width + j] = i+j; 
          }
      }

      // ********************************
      // create the forward and backward plans
      sig.forward2DFFT(data, spectrum, width, blockLength);
      sig.inverse2DFFT(spectrum, invertData, width, blockLength);

      // forward and inverse transforms
      sig.forward(data, spectrum);
      sig.inverse(spectrum, invertData);

      for (size_t i = 0; i< length; ++i){
          for (size_t j = 0; j< width; ++j){
              invertData[i*width + j] /=(width*length);
          }
      }

      double max_err_2DFFT = 0.0;
      double err = 0.0;
      for (size_t i = 0; i< length; ++i){
          for (size_t j = 0; j< width; ++j){
              err = std::abs(data[i*width + j] - invertData[i*width + j]);
              if (err > max_err_2DFFT)
                  max_err_2DFFT = err;
          }
      }
   
      // ********************************
    sig.forwardRangeFFT(data, spectrum, width, blockLength);
        sig.inverseRangeFFT(spectrum, invertData, width, blockLength);

        // forward and inverse transforms
        sig.forward(data, spectrum);
        sig.inverse(spectrum, invertData);

        for (size_t i = 0; i< length; ++i){
            for (size_t j = 0; j< width; ++j){
                invertData[i*width + j] /=(width);
            }
        }

        double max_err_range = 0.0;
        err = 0.0;
        for (size_t i = 0; i< length; ++i){
            for (size_t j = 0; j< width; ++j){
                err = std::abs(data[i*width + j] - invertData[i*width + j]);
                if (err > max_err_range)
                    max_err_range = err;
            }
        }

    // ************************************
    sig.forwardAzimuthFFT(data, spectrum, width, blockLength);
        sig.inverseAzimuthFFT(spectrum, invertData, width, blockLength);

        // forward and inverse transforms
        sig.forward(data, spectrum);
        sig.inverse(spectrum, invertData);

        for (size_t i = 0; i< length; ++i){
            for (size_t j = 0; j< width; ++j){
                invertData[i*width + j] /=(length);
            }
        }

        double max_err_az = 0.0;
        err = 0.0;
        for (size_t i = 0; i< length; ++i){
            for (size_t j = 0; j< width; ++j){
                err = std::abs(data[i*width + j] - invertData[i*width + j]);
                if (err > max_err_az)
                    max_err_az = err;
            }
        }
    // ************************************
    ASSERT_LT(max_err_2DFFT, 1.0e-12);

    ASSERT_LT(max_err_range, 1.0e-12);

    ASSERT_LT(max_err_az, 1.0e-12);
}


int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

