// Copyright (c) 2017-, California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged.
// All rights reserved.
// 
// Author(s):
// 
// !
//   \file BMFS.cc  -- Basic Math FunctionS 
//   \author Xiaoqing Wu

#include "BMFS.h"
#include "constants.h"

long long i_pow(int base, int power) {
  long long k = 1;
  for(int i = 0; i < power; i++) k *= (long long)base;
  return k;
}

int i_fftlength(int nr_pixels)
{
  int len = 1;
  while(1) {
    len *= 2;
    if(len >= nr_pixels) break;
  }
  return len;
}

int nint(float val)    //nearest integer
{
  return (int)(floor(val + 0.5));
}

int nint(double val)    //nearest integer
{
  return (int)(floor(val + 0.5));
}

int nint(long double val)    //nearest integer
{
  return (int)(floor(val + 0.5));
}

int i_floor(double val)    // floor integer
{
  return (int)(floor(val));
}

double square(double x)        // return x*x
{
  return x*x;
}

double cube(double x)          // return x*x*x
{
  return x*x*x;
}


int i_max(int a, int b) 
{
  if(a > b) return a;
  else return b;
}

int i_min(int a, int b)
{
  if(a < b) return a;
  else return b;
}

double d_max(double a, double b)
{
  if(a > b) return a;
  else return b;
}

double d_min(double a, double b)
{
  if(a < b) return a;
  else return b;
}


void moving_window_average(int items, double *input, int window_length, double ped)
{
  double *tmp = new double[items];
  for(int i = 0; i < items; i++) {
    tmp[i] = input[i];
  }

  double *weight = new double[2*window_length + 1];
  for(int i = 0; i < 2*window_length + 1; i++) {
    weight[i] = 0.5 + ped/2.0 - (0.5 - ped/2.0) * cos(PI * (double)i /(double)window_length);
  }

  for(int i = 0; i < items; i++) {
    double sum = 0;
    double count = 0;
    for(int ii = i - window_length; ii <= i + window_length; ii++) {
      if(ii < 0 || ii >= items) continue;
      double x = weight[ii - i + window_length];
      count += x;
      sum += tmp[ii]*x; 
    }
    if(count > 0) input[i] = sum/count;
  }
  delete[] tmp;
  delete[] weight;
}

/*
double *moving_window_average(int items, double *input, int window_length)
{
  double *out = new double[items];
  for(int i = 0; i < items; i++) {
    double sum = 0;
    int count = 0;
    for(int ii = i - window_length; ii <= i + window_length; ii++) {
      if(ii < 0 || ii >= items) continue;
      count ++;
      sum += input[ii]; 
    }
    if(count > 0) out[i] = sum/(double)count;
    else out[i] = input[i];
  }
  return out;
}
*/


double *standard_deviation(int items, double *input, int window_length, double *mean)
{
  double *dev = new double[items];
  for(int line = 0; line < items; line ++) {
    double sum = 0;
    int count = 0;
    double m = 0;
    for(int ii = line - window_length; ii <= line + window_length; ii ++) {
      if(ii < 0 || ii >= items) continue;
      if(mean) m = mean[ii];
      double x = input[ii] - m;
      sum += x*x;
      count ++;
    }
    if(count > 0) dev[line] = sqrt(sum/(double)count);
    else dev[line] = 0;
  }
  return dev;
}

D2point mean_std(int items, double *input)
{
  double mean = 0;
  double std = 0;
  for(int line = 0; line < items; line ++) {
    mean += input[line];
    std += input[line]*input[line];
  }
  mean /= (double) items;
  std = sqrt(std/(double)items - mean*mean);
  return D2point(mean, std);
}



double sinc(double x)
{
  if(fabs(x) < 1.0e-10) return 1.0;
  else return sin(x)/x;
}
