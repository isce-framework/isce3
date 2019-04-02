// Copyright (c) 2017-, California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged.
// All rights reserved.
// 
// Author(s):
// 
//  ======================================================================
// 
//  FILENAME: sort.h
// 
//  ======================================================================

#include "stdlib.h"

void	maxHeapify (double * vals, int n, int *indexes, int root);
void	maxHeapify (float * vals, int n, int *indexes, int root);
void	maxHeapify (int * vals, int n, int *indexes, int root);
void	heapSort (int n, double *val, int *indexes, int order = 0);  //   order == 0: small to large;  order == 1: large to small; 
void	heapSort (int n, float *val, int *indexes, int order = 0);  //   order == 0: small to large;  order == 1: large to small; 
void	heapSort (int n, int *val, int *indexes, int order = 0);  //   order == 0: small to large;  order == 1: large to small; 

void medium_filtering(int n, double *data, int win, double nodata);
void medium_filtering(int n, float *data, int win, float nodata);
float median_of_medians(int n, float *data, int rank);
float median_use_histogram(int n, float *data, int start_count = 0, int total_size = 0); 
// main call using 
// median_use_histogram(n, data, 0, n), 
// or simply 
// median_use_histogram(n, data);


