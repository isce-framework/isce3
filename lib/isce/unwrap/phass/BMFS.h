// Copyright (c) 2017-, California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged.
// All rights reserved.
// 
// Author(s):
// 
//  file BMFS.h  -- Basic Math FunctionS 
//  author Xiaoqing Wu

#ifndef BMFS_h
#define BMFS_h

#include <math.h>
#include <stdlib.h>
#include "Point.h"

long long i_pow(int base, int power);
int i_fftlength(int nr_pixels);
int nint(float val);            // nearest integer
int nint(double val);           // nearest integer
int nint(long double val);           // nearest integer
int i_floor(double val);        // floor integer

double square(double x);        // return x*x
double cube(double x);          // return x*x*x

int i_max(int a, int b);
int i_min(int a, int b);
double d_max(double d_a, double d_b);
double d_min(double d_a, double d_b);

D2point mean_std(int items, double *input);
void moving_window_average(int items, double *input, int window_length, double ped = 0.08); // ped: 1 for uniform; 0.5 Hanning; 0.08
//double *moving_window_average(int items, double *input, int window_length);

double *standard_deviation(int items, double *input, int window_length, double *mean = NULL);

double sinc(double x);

#endif

