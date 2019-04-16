// Copyright (c) 2017-, California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged.
// All rights reserved.
// 
// Author(s):
// 
//  ======================================================================
// 
//  FILENAME: PhaseStatistics.cc
// 
//  CREATED BY: Xiaoqing WU
// 
//  ======================================================================
#include "ASSP.h"
#include "PhaseStatistics.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define dphase_uppler_limit 0.5
#define dphase_lower_limit 0.01
using namespace std;


DataPatch<unsigned char> *compute_Hweight(int nr_lines, int nr_pixels, float **data)
{
  DataPatch<float> *Hdphase_patch = compute_H_delta_phase(nr_lines, nr_pixels, data);
  float **Hdphase = Hdphase_patch->get_data_lines_ptr();
  double upper_limit = dphase_uppler_limit; // 1.0;  // radians
  double lower_limit = dphase_lower_limit;  // radians

  
  DataPatch<unsigned char> *Hweight_patch = new DataPatch<unsigned char>(nr_pixels, nr_lines - 1);
  unsigned char **Hweight = Hweight_patch->get_data_lines_ptr();
  for(int line = 0; line < nr_lines - 1; line ++) {
    for(int pixel = 0; pixel < nr_pixels; pixel ++) {
      double x = fabs(Hdphase[line][pixel]);
      if(x < lower_limit) x = lower_limit;
      if(x > upper_limit) x = upper_limit;
      Hweight[line][pixel] = 255 - 254 * (x - lower_limit)/(upper_limit - lower_limit);
    }
  }
  delete Hdphase_patch;
  return Hweight_patch;
}

DataPatch<unsigned char> *compute_Vweight(int nr_lines, int nr_pixels, float **data)
{
  DataPatch<float> *Vdphase_patch = compute_V_delta_phase(nr_lines, nr_pixels, data);
  float **Vdphase = Vdphase_patch->get_data_lines_ptr();

  double upper_limit = dphase_uppler_limit; // 1.0;  // radians
  double lower_limit = dphase_lower_limit;  // radians
  
  DataPatch<unsigned char> *Vweight_patch = new DataPatch<unsigned char>(nr_pixels - 1, nr_lines);
  unsigned char **Vweight = Vweight_patch->get_data_lines_ptr();
  for(int line = 0; line < nr_lines; line ++) {
    for(int pixel = 0; pixel < nr_pixels - 1; pixel ++) {
      double x = fabs(Vdphase[line][pixel]);
      if(x < lower_limit) x = lower_limit;
      if(x > upper_limit) x = upper_limit;
      Vweight[line][pixel] = 255 - 254 * (x - lower_limit)/(upper_limit - lower_limit);
    }
  }
  delete Vdphase_patch;
  return Vweight_patch;
}

DataPatch<float> *compute_H_delta_phase(int nr_lines, int nr_pixels, float **data)
{
  DataPatch<float> *delta_line_patch = new DataPatch<float>(nr_pixels, nr_lines - 1);
  float **delta_line = delta_line_patch->get_data_lines_ptr();

  double pi = PI;
  double two_pi = 2.0 * pi;
 
  float *tmp_data = new float[nr_pixels];

  float void_phase = -10000.0;

  int win = 1;
//# pragma omp parallel for
  for(int line = 0; line < nr_lines - 1; line ++) {
    for(int pixel = 0; pixel < nr_pixels; pixel ++) {
      double x = data[line + 1][pixel] - data[line][pixel];
      if(x > pi)  x -= two_pi;
      if(x < -pi) x += two_pi;
      delta_line[line][pixel] = x;
      tmp_data[pixel] = x;
    }
    for(int pixel = 0; pixel < nr_pixels; pixel ++) {
      int count = 0;
      double sum = 0;
      double check_sum = 0;
      for(int ii = pixel - win; ii <= pixel + win; ii++) {
        if(ii < 0 || ii >= nr_pixels) continue;
	if(tmp_data[ii] == void_phase) continue;
	check_sum += fabs(tmp_data[ii]);
	count ++;
	sum += tmp_data[ii];
      }
      if(count == 0 || check_sum == 0) delta_line[line][pixel] = PI;
      else delta_line[line][pixel] = sum/(double)count;
    }
  }
  delete[] tmp_data;
  return delta_line_patch;
}


DataPatch<float> *compute_V_delta_phase(int nr_lines, int nr_pixels, float **data)
{
cerr << "compute_V_delta_phase() ...... " <<  endl;
  DataPatch<float> *delta_pixel_patch = new DataPatch<float>(nr_pixels - 1, nr_lines);
  float **delta_pixel = delta_pixel_patch->get_data_lines_ptr();

  double pi = PI;
  double two_pi = 2.0 * pi;
 
  float *tmp_data = new float[nr_lines];

  float void_phase = -10000.0;
  

  int win = 1;
//# pragma omp parallel for

  for(int pixel = 0; pixel < nr_pixels - 1; pixel ++) {
    for(int line = 0; line < nr_lines; line ++) {
      double x = data[line][pixel + 1] - data[line][pixel];
      if(x > pi)  x -= two_pi;
      if(x < -pi) x += two_pi;
      delta_pixel[line][pixel] = x;
      tmp_data[line] = x;
    }

    for(int line = 0; line < nr_lines; line ++) {
      int count = 0;
      double sum = 0;
      double check_sum = 0;
      for(int ii = line - win; ii <= line + win; ii++) {
        if(ii < 0 || ii >= nr_lines) continue;
	if(tmp_data[ii] == void_phase) continue;
	check_sum += fabs(tmp_data[ii]);

	count ++;
	sum += tmp_data[ii];
      }
      if(count == 0 || check_sum == 0) delta_pixel[line][pixel] = PI;
      else delta_pixel[line][pixel] = sum/(double)count;
    }
  }
  delete[] tmp_data;
  return delta_pixel_patch;
}


void compute_corr(int nr_lines, int nr_pixels, float **data, unsigned char **corr_data, double max_phase_std)
{
  DataPatch<float> *delta_pixel_patch = new DataPatch<float>(nr_pixels, nr_lines);
  float **delta_pixel = delta_pixel_patch->get_data_lines_ptr();
//  DataPatch<float> *delta_pixel_mean_patch = new DataPatch<float>(nr_pixels, nr_lines);
//  float **delta_pixel_mean = delta_pixel_mean_patch->get_data_lines_ptr();

  DataPatch<float> *delta_line_patch = new DataPatch<float>(nr_pixels, nr_lines);
  float **delta_line = delta_line_patch->get_data_lines_ptr();
//  DataPatch<float> *delta_line_mean_patch = new DataPatch<float>(nr_pixels, nr_lines);
//  float **delta_line_mean = delta_line_mean_patch->get_data_lines_ptr();

  DataPatch<float> *std_patch = new DataPatch<float>(nr_pixels, nr_lines);
  float **std_data = std_patch->get_data_lines_ptr();

  double pi = PI;
  double two_pi = 2.0 * pi;

# pragma omp parallel for
  for(int line = 0; line < nr_lines; line ++) {
    for(int pixel = 0; pixel < nr_pixels; pixel ++) {
      delta_pixel[line][pixel] = 0;
      delta_line[line][pixel] = 0;
//      delta_pixel_mean[line][pixel] = 0;
//      delta_line_mean[line][pixel] = 0;
      std_data[line][pixel] = 0;
    }
  }

# pragma omp parallel for
  for(int line = 0; line < nr_lines; line ++) {
    for(int pixel = 0; pixel < nr_pixels - 1; pixel ++) {
      double x = data[line][pixel + 1] - data[line][pixel];
      if(x > pi)  x -= two_pi;
      if(x < -pi) x += two_pi;
      delta_pixel[line][pixel] = x;
    }
  }

  int win = 1;
  int w2 = (2*win + 1)*(2*win + 1);

# pragma omp parallel for
  for(int line = 0; line < nr_lines - 1; line ++) {
    for(int pixel = 0; pixel < nr_pixels - 1; pixel ++) {
      double mean = 0;
      int count = 0;
      for(int ii = line - win; ii <= line + win; ii++) {
        if(ii < 0 || ii >= nr_lines - 1) continue;
	for(int jj = pixel - win; jj <= pixel + win; jj++) {
	  if(jj < 0 || jj >= nr_pixels - 1) continue;
	  count ++;
	  mean += delta_pixel[ii][jj];
	}
      }
      if(count == w2) mean /= (double)count;
//      delta_pixel_mean[line][pixel] = mean;
      
      double dev = 0;
      for(int ii = line - win; ii <= line + win; ii++) {
        if(ii < 0 || ii >= nr_lines - 1) continue;
	for(int jj = pixel - win; jj <= pixel + win; jj++) {
	  if(jj < 0 || jj >= nr_pixels - 1) continue; 
          double x = delta_pixel[ii][jj];// - mean;
	  dev += x*x;
	}
      }
      if(count == w2) dev /= (double)count;
      std_data[line][pixel] = dev;
    }
  }

# pragma omp parallel for
  for(int line = 0; line < nr_lines - 1; line ++) {
    for(int pixel = 0; pixel < nr_pixels; pixel ++) {
      double x = data[line + 1][pixel] - data[line][pixel];
      if(x > pi)  x -= two_pi;
      if(x < -pi) x += two_pi;
      delta_line[line][pixel] = x;
    }
  }

# pragma omp parallel for
  for(int line = 0; line < nr_lines - 1; line ++) {
    for(int pixel = 0; pixel < nr_pixels - 1; pixel ++) {
      double mean = 0;
      int count = 0;
      for(int ii = line - win; ii <= line + win; ii++) {
        if(ii < 0 || ii >= nr_lines - 1) continue;
	for(int jj = pixel - win; jj <= pixel + win; jj++) {
	  if(jj < 0 || jj >= nr_pixels - 1) continue;
	  count ++;
	  mean += delta_line[ii][jj];
	}
      }
      if(count == w2) mean /= (double)count;
//      delta_line_mean[line][pixel] = mean;
      
      double dev = 0;
      for(int ii = line - win; ii <= line + win; ii++) {
        if(ii < 0 || ii >= nr_lines - 1) continue;
	for(int jj = pixel - win; jj <= pixel + win; jj++) {
	  if(jj < 0 || jj >= nr_pixels - 1) continue; 
          double x = delta_line[ii][jj];// - mean;
	  dev += x*x;
	}
      }
      if(count == w2) dev /= (double)count;
      std_data[line][pixel] += dev;
      std_data[line][pixel] /= 4.0;
    }
  }

  double phi_max = pi/2.0;

  double scale = 0.025; // riginal value 0.05
# pragma omp parallel for

  for(int line = 0; line < nr_lines; line ++) {
    for(int pixel = 0; pixel < nr_pixels; pixel ++) {
      if(std_data[line][pixel] == 0) corr_data[line][pixel] = (unsigned char)cost_scale;
      else {
        //double x = delta_pixel[line][pixel];// - delta_pixel_mean[line][pixel];
        //double y = delta_line[line][pixel];// - delta_pixel_mean[line][pixel];
        //x = 0.5*(x*x + y*y)/phi_max/phi_max/(std_data[line][pixel] + 0.1);
        //x = std_data[line][pixel];
        double x = scale/std_data[line][pixel];
        if(x > 1.0) x = 1.0;
        //if(x == 0) continue;
        //if(x < 0.1) x = 0.1;
        corr_data[line][pixel] = (unsigned char)(cost_scale * x);
      }
    }
  }

//  cerr << "cost_scale: " << cost_scale << endl;

/*
  for(int line = 0; line < nr_lines; line ++) {
    for(int pixel = 0; pixel < nr_pixels; pixel ++) {
      if(std_data[line][pixel] == 0) corr_data[line][pixel] = 0;
      else {
        double x = (max_phase_std - std_data[line][pixel])/max_phase_std;
        if(x < 0) x = 0; 
        corr_data[line][pixel] = (unsigned char)(cost_scale * x);
      }
    }
  }
*/
  
  delete std_patch;
  delete delta_line_patch;
  delete delta_pixel_patch;
//  delete delta_line_mean_patch;
//  delete delta_pixel_mean_patch;
}

