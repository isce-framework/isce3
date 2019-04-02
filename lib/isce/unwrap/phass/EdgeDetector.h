// Copyright (c) 2017-, California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged.
// All rights reserved.
// 
// Author(s):
// 
//  ======================================================================
// 
//  FILENAME: EdgeDetector.h
//   
//  CREATED BY: Xiaoqing WU
// 
//  ======================================================================

#ifndef  EdgeDetector_h
#define  EdgeDetector_h

#include "Point.h"
#include "constants.h"
#include "DataPatch.h"


void detect_edge(int nr_lines, int nr_pixels, float **data, unsigned char **edge_data,
		 int window_length, double C_min, double R_edge, double R_line);

void detect_edge(int nr_lines, int nr_pixels, float **data, 
		 unsigned char **horizontal_edge_data, unsigned char **vertical_edge_data,
		 int window_length, double coefficient_variance_min, double max_edge_ratio);

#endif

