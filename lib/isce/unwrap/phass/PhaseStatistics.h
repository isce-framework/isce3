// Copyright (c) 2017-, California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged.
// All rights reserved.
// 
// Author(s):
// 
//  ======================================================================
// 
//  FILENAME: PhaseStatistics.h
//   
//  CREATED BY: Xiaoqing WU
// 
//  ======================================================================

#ifndef  PhaseStatistics_h
#define  PhaseStatistics_h

#include "Point.h"
#include "constants.h"
#include "DataPatch.h"


void compute_corr(int nr_lines, int nr_pixels, float **data, unsigned char **corr_data, double max_phase_std);
DataPatch<unsigned char> *compute_Hweight(int nr_lines, int nr_pixels, float **data);
DataPatch<unsigned char> *compute_Vweight(int nr_lines, int nr_pixels, float **data);
DataPatch<float> *compute_H_delta_phase(int nr_lines, int nr_pixels, float **data);
DataPatch<float> *compute_V_delta_phase(int nr_lines, int nr_pixels, float **data);


#endif

