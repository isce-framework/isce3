// Copyright (c) 2017-, California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged.
// All rights reserved.
// 
// Author(s):
// 

#include <complex>

#ifndef LSPD
#define LSPD 299792458.0L
#endif


#ifndef PI
#define PI 3.141592653589793L
#endif

#ifndef BOLTZDB
#define BOLTZDB -228.601209136
#endif

#ifndef LN2
#define LN2 0.69314718055994529
#endif

#ifndef DEG2RAD
#define DEG2RAD 0.017453292519943295
#endif

#ifndef EARTH_RADIUS
#define EARTH_RADIUS 6378137.0
#endif

#ifndef EARTH_E2
#define EARTH_E2  0.0066943799901    /* 0.00669438 */
#endif

#define EARTH_G		((double) 6.6622000E-11) /* gravitational const of Earth */
#define EARTH_MASS	((double) 5.974E24)	 /* earth mass */

#define OUT_OF_BOUNDS -999.

#ifndef H_TRK_NULL
#define H_TRK_NULL -99.
#endif

#ifndef MASK_TRK_NULL 
#define MASK_TRK_NULL 255
#endif

#define SRNG_NULL -1
#define FIRST 0

// These are used to interpret the mask
#define LAND  0
#define WATER 1
#define NO_CLASS 255
#define LAYOVER_NULL -99


#define PTR_SCALE 1.8789e8 // scale to make area under PTR = 1 for Gaussian PTR
                             
// these are for use by classification

#define NOT_UNIFORM -1
#define LAND_UNIFORM 0
#define WATER_UNIFORM 1
#define NULL_WEIGHT complex<double> (0.,0.)
#define POWER_NULL  1.
#define INTF_IMAG_NULL  0.
#define INTF_REAL_NULL  0.
#define T0 290.

#define MAX_CHAN  2         // Total channels to process
#define MAX_CPDEG 2         // maximal degree of caltone phase polynomial fit
#define MAXCHEBY  1000      // maximal order of Chebyshev polynomial
#define MAX_PHSDRIFTORDER 5 // maximal order of phase drift polynomial
#define MAX_RNGFFT 65536    // maximal order of phase drift polynomial
#define MAX_RNGSAMPS 10000  // maximum valid range samples
#define MAX_NOTCHES 10      // 
#define WVL_DEG 2           // maximal degree of wavelength fit
#define MAX_RNGBYTES   65536 

#define STD_TVPPARAMS  15
#define MAX_LEVERS     6
#define MAX_GAINS      6
#define MAX_CALPARAMS  10
#define MAX_TVPPARAMS  STD_TVPPARAMS+MAX_GAINS+MAX_LEVERS+MAX_CALPARAMS+1

#define FIXED_TVPREAD  15
#define ESA            1
#define LEGACY         2

#define TVPF_HdrRecs   2
#define TVPF_RECLEN    2

#define MAX_CALPUL     250
#define MAXINTKERLGH   21
#define BLKMULT        12
#define BLKSIZE       252

#define TD_CALCHIRP     2.2222040e-6
#define WINSIZ_CALCHIRP 90.0e-6
#define CALSRCH 8

//#define FORWARD_FFT   -1
//#define INVERSE_FFT    1

#define NULL_WIN   21

#define GAINREF 2.51188643151e-2

#define ANTENNA_ANGLE_DELTA 0.02       // deg
#define ANTENNA_AZIMUTH_START -20.0    // deg
#define ANTENNA_AZIMUTH_NR_POINTS 2001
#define ANTENNA_AZIMUTH_BIAS -0.28     // deg (peak position)
#define ANTENNA_RANGE_START -60.0      // deg
#define ANTENNA_RANGE_NR_POINTS 6001
#define ANTENNA_RANGE_BIAS -13.235       // deg (peak position)


#define i_active    1
#define i_notactive 0
#define i_calibrationmode 2

#define i_on  1
#define i_off 0
#define i_awg      1
#define i_analytic 0
#define i_byte     2
#define i_floating 4

#define i_bfpq    1
#define i_8bit    2
#define i_adc_bits 8
//#define i_blkl    128
#define i_fbpq_block_length    128
#define i_ham     1
#define i_noham   0

#define i_sat      1
#define i_pingpong 2

#define i_iq_quantization_scheme_factor  2
#define i_lowerband  -1
#define i_upperband   1

#define i_offsetvideo 1

#define  i_left   1
#define  i_right -1

#define i_ant1ch  0
#define i_ant2ch  1
#define i_ant3ch  2

#define fcomplex complex<float>

#define r_relfiltlenps 6.0
#define r_betaps       0.85
#define r_pedestalps   1.0
#define i_decfactorps  8192
#define i_blocksizeps  128

#define i_sinc_length_bp 8
#define r_beta_bp        0.85
#define r_pedestal_bp    0.0
#define i_decfactor_bp   8192

#define PULSELENGTHREF 40.0e-6

#define SFACT  4000.0
#define PSWF_A_LOW 1.5029
#define PSWF_B_LOW 3.141592654
#define PSWF_A 1.5029
#define PSWF_B 3.141592654
#define PSWF_A_HIGH 5.2630
#define PSWF_B_HIGH 0.000001

#define Null_Height_Value -10000.0
#define Null_Phase_Value  -10000.0
#define Null_Amplitude_Value 0.0
#define Null_Height_Error_Value -1.0
#define Null_Correlation_Value -1.0

#define AirSWOT_lrl 1
//#define AirSWOT_center_frequency 35.852e9 // 35.75e9
//#define AirSWOT_Roll_Bias   0.0   // degree
//#define AirSWOT_Yaw_Bias    -2.23  // -2.55  // -1.87  //-1.513 // -1.4922
//#define AirSWOT_Pitch_Bias  1.0  //1.02 // 0.95 //0.85  // 0.733 //0.8245

#define AirSWOT_antenna_panel_roll  0.0 // 0.016 // -0.2690  // degree
#define AirSWOT_antenna_panel_pitch 0.0 // 0.0746 //  0.4838  // degree
#define AirSWOT_antenna_panel_yaw   0.0 // -1.3 //-0.6942 // -0.2981  // degree

#define AirSWOT_azimuth_beamwidth 2.0  // degree
//#define AirSWOT_profile_repetition_time 192.8e-6 // 144.6e-6  // second

#define AirSWOT_far_range_lookangle 30.0  // 25.0  // degrees


#define Glistin_azimuth_beamwidth 0.8  // degree
#define Glistin_antenna_panel_roll  0.0 //  degree
#define Glistin_antenna_panel_pitch 0.0 //  degree
#define Glistin_antenna_panel_yaw   0.0 //  degree
