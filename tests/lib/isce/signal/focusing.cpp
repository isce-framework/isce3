
#include <iostream>
#include <cstdio>
#include <sstream>
#include <fstream>
#include <cmath>
#include <complex>
#include <gtest/gtest.h>

#include "isce/signal/Signal.h"
#include "isce/io/Raster.h"
#include "isce/signal/Filter.h"

  //TEST(Crossmul, InterferogramZero)
  //{
int main()
{
    
    // B.) Sensor parameters (ERS satellite)
    double fs = 18.962468*std::pow(10.0,6.0);         // Range Sampling Frequency [Hz]
    double K_r = 4.18989015*std::pow(10.0,11.0);      // FM Rate Range Chirp [1/s^2] --> up-chirp
    double tau_p = 37.12*std::pow(10,-6);     // Chirp duration [s]
    double V = 7098.0194;                // Effective satellite velocity [m/s]
    double Lambda = 0.05656;             // Length of carrier wave [m]
    double R_0 = 852358.15;              // Range to center of antenna footprint [m]
    double ta = 0.6;                     // Aperture time [s]
    double prf = 1679.902;               // Pulse Repitition Frequency [Hz]

    //isce::io::Raster Raw_data_raster("/Users/fattahi/Tutorials/ID_1302_SAR-EDU_Tutorial_Focusing_Python/Data/test.bin.vrt");
    isce::io::Raster Raw_data_raster("/Users/fattahi/Tutorials/ID_1302_SAR-EDU_Tutorial_Focusing_Python/Data/ERS.raw.vrt");

    size_t size_azimuth = Raw_data_raster.length();
    size_t size_range = Raw_data_raster.width();

    std::cout << "size_azimuth: " << size_azimuth << std::endl;
    std::cout << "size_range: " << size_range << std::endl;

    std::valarray<std::complex<double>> Raw_data(size_azimuth*size_range);
    std::valarray<std::complex<double>> range_chirp(size_azimuth*size_range);

    std::cout << "get a block of data" << std::endl;
    Raw_data_raster.getBlock(Raw_data, 0, 0, size_range, size_azimuth);

    std::cout << "tau , omega" << std::endl;
    //size_t size_chirp_r = 704;
    size_t size_chirp_r = std::ceil(tau_p*fs);

    std::valarray<double> tau(size_chirp_r); // time in range
    std::valarray<double> omega(size_chirp_r); // frequency in range
    /*for (size_t i; i<tau.size(); i++){
        tau[i] = -1.0*tau_p/2.+i/fs;
        omega[i] = -1.0*fs/2.0+i/tau_p;
    }*/

    for (size_t i; i<tau.size(); i++){
          tau[i] = -1.0*tau_p/2.+i/fs;
          omega[i] = -1.0*fs/2.0+i/tau_p;
    }
    

    std::cout << "ra_chirp_temp" << std::endl;
    std::valarray<std::complex<double>> ra_chirp_temp(size_chirp_r);   
    for (size_t i; i<tau.size(); i++){
        //we actually want the conjugate of the chirp
        double phase = M_PI*K_r*std::pow(tau[i],2);
        ra_chirp_temp[i] = std::complex<double>(std::cos(phase), std::sin(phase));
    }
    
    std::cout << M_PI*K_r*std::pow(tau[0],2) << std::endl;
    std::cout << M_PI*K_r*std::pow(tau[100],2) << std::endl;
    std::cout << M_PI*K_r*std::pow(tau[500],2) << std::endl;
    std::cout << ra_chirp_temp[0] << std::endl;
    std::cout << ra_chirp_temp[100] << std::endl;
    std::cout << ra_chirp_temp[500] << std::endl;
    
    // Position chirp in range vector (centered)
    // used for correlation procedure
    std::cout << "ra_chirp" << std::endl;
    size_t index_start = std::ceil((size_range-size_chirp_r)/2)-1;
    size_t index_end = size_chirp_r + std::ceil((size_range-size_chirp_r)/2)-2;
    std::cout<< "index_start, index_end" << index_start << ", "<< index_end << std::endl;
    for (size_t line=0; line<size_azimuth; ++line){
        for (size_t i=0; i<ra_chirp_temp.size(); ++i){
            range_chirp[line*size_range + index_start + i] = ra_chirp_temp[i];
        }
    }
     
    std::valarray<std::complex<double>> dataSpectrum(size_azimuth*size_range);
    std::valarray<std::complex<double>> RANGE_CHIRP(size_azimuth*size_range);   

    isce::signal::Signal<double> sig;
    sig.forwardRangeFFT(range_chirp, RANGE_CHIRP, size_range, size_azimuth, size_range, size_azimuth );
    sig.inverseRangeFFT(RANGE_CHIRP, range_chirp, size_range, size_azimuth, size_range, size_azimuth);

    std::cout << "forward fft" << std::endl;
    sig.forward(range_chirp, RANGE_CHIRP);

    //conjugate of the RANGE_CHIRP
    for (size_t i = 0 ; i< RANGE_CHIRP.size(); ++i){
        RANGE_CHIRP[i] = std::conj(RANGE_CHIRP[i]);
    }

    std::cout << "forward fft data" << std::endl;
    sig.forward(Raw_data, dataSpectrum);
 
    std::valarray<std::complex<double>> rangeCompressed(size_azimuth*size_range);
    
    dataSpectrum *= RANGE_CHIRP; 
    sig.inverse(dataSpectrum, rangeCompressed); 

    rangeCompressed /=size_range;

    isce::io::Raster rngCompRaster("test.rngcompressed", size_range, size_azimuth, 1, GDT_CFloat64,   "ENVI");
    std::cout << "set block" << std::endl;
    rngCompRaster.setBlock(rangeCompressed, 0, 0, size_range, size_azimuth);
    
    
    //isce::io::Raster rng_comp_raster("/Users/fattahi/Tutorials/ID_1302_SAR-EDU_Tutorial_Focusing_Python/Executables/rangecompressed.bin.vrt");

    //rng_comp_raster.getBlock(rangeCompressed, 0, 0, size_range, size_azimuth);

    //std::cout << "rangeCompressed[973583] : "<< rangeCompressed[973583] << std::endl;
    

    // Azimuth chirp
    std::valarray<std::complex<double>> azimuth_chirp(size_azimuth*size_range);
    //size_t size_chirp_a = 1008;
    size_t size_chirp_a = std::ceil(ta*prf);
    std::valarray<double> t(size_chirp_a); // time in azimuth
    std::valarray<double> v(size_chirp_a); // frequency in azimuth
    for (size_t i=0; i<t.size(); i++){
        t[i] = -1.0*ta/2.+i/prf;
        v[i] = -1.0*prf/2.0+i/ta;
    }
    
    // FM rate Azimuth Chirp
    double K_a = (-2*std::pow(V,2))/(Lambda*R_0);
    std::cout << "az_chirp_temp" << std::endl;
    std::valarray<std::complex<double>> az_chirp_temp(size_chirp_a);
    std::cout << std::setprecision(16)<< "t: " << t[100] << std::endl;
    std::cout << K_a << std::endl;

    for (size_t i=0; i<t.size(); i++){
          double phase = M_PI*K_a*std::pow(t[i],2);
          if (i==100)
                std::cout << "phase " << phase << std::endl;          
          az_chirp_temp[i] = std::complex<double>(std::cos(phase), std::sin(phase));
    }   
    std::cout << "azimuth_chirp: " << az_chirp_temp[0] << std::endl;
    std::cout << "azimuth_chirp: " << az_chirp_temp[100] << std::endl;
    std::cout << "azimuth_chirp: " << az_chirp_temp[200] << std::endl;
    int index_start2 = std::ceil((size_azimuth-size_chirp_a)/2)-1;
    int index_end2 = size_chirp_a+std::ceil((size_azimuth-size_chirp_a)/2)-2;

    //each column of the block is the azimuth chirp
    //for (size_t line=index_start2; line<index_end2; ++line){
    for (size_t line= 0 ; line<az_chirp_temp.size(); ++line){
    	for (size_t col=0; col<size_range; ++col){
            azimuth_chirp[(line+index_start2)*size_range + col] = az_chirp_temp[line];
    	}
    }

    std::cout << "azimuth_chirp: " << azimuth_chirp[size_range*index_start2] << std::endl;
    std::cout << "azimuth_chirp: " << azimuth_chirp[size_range*(100+index_start2)]  << std::endl;
    std::cout << "azimuth_chirp: " << azimuth_chirp[size_range*(200+index_start2)]<< std::endl;

    //std::valarray<std::complex<double>> dataSpectrum(size_azimuth*size_range);
    std::valarray<std::complex<double>> AZIMUTH_CHIRP(size_azimuth*size_range);
    
    isce::signal::Signal<double> sigAz;
    sigAz.forwardAzimuthFFT(azimuth_chirp, AZIMUTH_CHIRP, size_range, size_azimuth, size_range, size_azimuth );
    sigAz.inverseAzimuthFFT(AZIMUTH_CHIRP, azimuth_chirp, size_range, size_azimuth, size_range, size_azimuth);

    std::cout << "forward fft" << std::endl;
    sigAz.forward(azimuth_chirp, AZIMUTH_CHIRP);
    for (size_t i = 0 ; i< AZIMUTH_CHIRP.size(); ++i){
          AZIMUTH_CHIRP[i] = std::conj(AZIMUTH_CHIRP[i]);
    }

    std::cout << "AZIMUTH_CHIRP: " << AZIMUTH_CHIRP[size_range*index_start2] << std::endl;
    std::cout << "AZIMUTH_CHIRP: " << AZIMUTH_CHIRP[size_range*(100+index_start2)]  << std::endl;
    std::cout << "AZIMUTH_CHIRP: " << AZIMUTH_CHIRP[size_range*(200+index_start2)]<< std::endl;

    std::cout << "rangeCompressed: " << rangeCompressed[size_range*index_start2] << std::endl;
    std::cout << "rangeCompressed: " << rangeCompressed[size_range*(100+index_start2)]  << std::endl;
    std::cout << "rangeCompressed: " << rangeCompressed[size_range*(200+index_start2)]<< std::endl;


    std::cout << "forward fft data" << std::endl;
    sigAz.forward(rangeCompressed, dataSpectrum);

    std::cout << "dataSpectrum: " << dataSpectrum[size_range*index_start2] << std::endl;
    std::cout << "dataSpectrum: " << dataSpectrum[size_range*(100+index_start2)]  << std::endl;
    std::cout << "dataSpectrum: " << dataSpectrum[size_range*(200+index_start2)]<< std::endl;


    std::valarray<std::complex<double>> azCompressed(size_azimuth*size_range);

    dataSpectrum *= AZIMUTH_CHIRP;

    std::cout << "dataSpectrum after multiply by chirp" << std::endl;

    std::cout << "dataSpectrum: " << dataSpectrum[size_range*index_start2] << std::endl;
    std::cout << "dataSpectrum: " << dataSpectrum[size_range*(100+index_start2)]  << std::endl;
    std::cout << "dataSpectrum: " << dataSpectrum[size_range*(200+index_start2)]<< std::endl;

    sigAz.inverse(dataSpectrum, azCompressed);

    azCompressed /= size_azimuth;

    std::cout << "azCompressed: " << azCompressed[size_range*index_start2] << std::endl;
    std::cout << "azCompressed: " << azCompressed[size_range*(100+index_start2)]  << std::endl;
    std::cout << "azCompressed: " << azCompressed[size_range*(200+index_start2)]<< std::endl;


    isce::io::Raster azCompRaster("test.slc", size_range, size_azimuth, 1, GDT_CFloat64,   "ENVI");
    std::cout << "set block" << std::endl;
    azCompRaster.setBlock(azCompressed, 0, 0, size_range, size_azimuth);

    return 0;
}

