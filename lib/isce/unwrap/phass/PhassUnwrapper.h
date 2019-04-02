// Copyright (c) 2017-, California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged.
// All rights reserved.
// 
// Author(s):
// 
//  ======================================================================
// 
//  FILENAME: PhassUnwrapper.h
//   
//  CREATED BY: Xiaoqing WU
// 
//  ======================================================================


#ifndef  PhassUnwrapper_h
#define  PhassUnwrapper_h

void phass_unwrap(int nr_lines, int nr_pixels, float **phase_data, float **corr_data, float **power, int **region_map,
		  double corr_th, double good_corr, int min_pixels_per_region);

#endif

