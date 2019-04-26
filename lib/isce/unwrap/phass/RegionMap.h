// Copyright (c) 2017-, California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged.
// All rights reserved.
// 
// Author(s):
// 
//  ======================================================================
// 
//  FILENAME: RegionMap.h
//   
//  CREATED BY: Xiaoqing WU
// 
//  ======================================================================

#ifndef  RegionMap_h
#define  RegionMap_h

#include "Point.h"
#include "constants.h"

#include <list>
#include <set>
#include <queue>

#define face_up     0x01
#define face_down   0x02
#define face_left   0x04
#define face_right  0x08


using namespace std;

int create_region_map(int lines, int pixels, float **phases, int **region_map, int min_pixels_per_region = 0, float nodata = -10000.0);
int create_region_map(int lines, int pixels, bool **mask, int **region_map, int min_pixels_per_region = 0);
int create_region_map(int lines, int pixels, bool **mask, unsigned char **disc_data, int **region_map, int min_pixels_per_region);
int create_region_map(int lines, int pixels, bool **mask, float **normalized_heights, int **region_map, int min_pixels_per_region);
int create_region_map(int lines, int pixels, int **input_regions, float **normalized_heights, int **region_map, int min_pixels_per_region);
int create_region_map(int lines, int pixels, int **input_regions, int **region_map, int min_pixels_per_region);
// Inputs: lines, pixels, bool_data, region_map
// Output: region_map, nr_regions, region_points

list<USPoint> *make_point_list(int lines, int pixels, int **region_map, int nr_regions);
// Inputs: lines, pixels, region_map, nr_regions
// Output: list array with each list


void make_point_list(int lines, int pixels, int **region_map, int &nr_regions, list<USPoint> **point_list);

#endif
