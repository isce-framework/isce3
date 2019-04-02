// Copyright (c) 2017-, California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged.
// All rights reserved.
// 
// Author(s):
// 
//  ----------------------------------------------------------------------------
//  Author:  Xiaoqing Wu
// 

#include "ChangeDetector.h"

void ChangeDetector::basic_init()
{
  nr_lines = 0;
  nr_pixels = 0;
  no_data = 0;
  change_type = 0;
  change_th = 0;
  window_size = 0;
  iterations = 0;
  max_change = 0;

  data_patch = NULL;
  change_patch = NULL;
}

ChangeDetector::ChangeDetector(int nr_lines_, int nr_pixels_, float no_data_, DataPatch<float> *data_patch_, 
		 int change_type_, float change_th_, int window, int iter)
{
  basic_init();
  nr_lines = nr_lines_;
  nr_pixels = nr_pixels_;
  no_data = no_data_;
  data_patch = data_patch_;
  change_type = change_type_;
  change_th = change_th_;
  window_size = window;
  iterations = iter;

  change_patch = new DataPatch<unsigned char>(nr_pixels, nr_lines);
  unsigned char **change_data = change_patch->get_data_lines_ptr();
  for(int i = 0; i < nr_lines; i ++) {
    for(int j = 0; j < nr_pixels; j++) {
      change_data[i][j] = 0;
    }
  }
  calculate();
}

ChangeDetector::~ChangeDetector() 
{
  delete change_patch;
}

void ChangeDetector::calculate()
{
  float **data = data_patch->get_data_lines_ptr();
  unsigned char **change_data = change_patch->get_data_lines_ptr();

  int box_size = window_size;
  cerr << "box_size: " << box_size << endl;

  for(int iter = 0; iter < iterations; iter++) {

    int N = 1 + nr_lines - box_size;
    int M = 1 + nr_pixels - box_size;

    cerr << "iter: " << iter << "  box_size: " << box_size << endl;
//  cerr << "N: (line direction) " << N << endl;
//  cerr << "M: (pixel direction) " << M << endl;

// # pragma omp parallel for 
    for(int n = 0; n < N; n ++) {
      DataPatch<float> *work_patch = new DataPatch<float>(box_size, box_size);
      float **work_data = work_patch->get_data_lines_ptr();
    
      int start_line = n;
      for(int m = 0; m < M; m ++) {
        int start_pixel = m;
        for(int i = 0; i < box_size; i ++) {
	  for(int j = 0; j < box_size; j++) {
	    work_data[i][j] = data[start_line + i][start_pixel + j];
	  }
	} 

        int count = 0;           
  
 	double maxv = -1.0e20;
        double minv =  1.0e20;
  
        for(int i = 0; i < box_size; i++) {
          for(int j = 0; j < box_size; j++) {
	    float x = work_data[i][j];
	    if(x == no_data) continue;
	    count ++;
	    if(x > maxv) maxv = x;
	    if(x < minv) minv = x;
	  }
	}
        if(count < box_size) continue;

        double delta = (maxv - minv)/(double)(2*box_size);
        int items = (maxv - minv)/delta + 1;


        int *histo = new int[items];
        for(int i = 0; i < items; i++) histo[i] = 0;
        bool *checked = new bool[items];
        for(int i = 0; i < items; i++) checked[i] = false;

        for(int i = 0; i < box_size; i++) {
          for(int j = 0; j < box_size; j++) {
	    double x = work_data[i][j];
	    if(x == no_data) continue;
	    int k = (x - minv)/delta;
            if(k < items) histo[k] ++;
	  }
	}

        int peak1 = 0;
        int peak1_id = 0;
        for(int ii = 0; ii < items; ii ++) {
          if(histo[ii] > peak1) {
	    peak1 = histo[ii];
	    peak1_id = ii;
          }
        }
        checked[peak1_id] = true;
//        if(n == 0 && m == 60) cerr << "peak1: " << peak1 << "  peak1_id: " << peak1_id << "  checked: " << (int)checked[peak1_id] << endl;
        
        for(int ii = peak1_id + 1; ii < items; ii ++) {
          if(histo[ii] <= histo[ii - 1]) {

//if(n == 0 && m == 60) cerr << " +++++   ii: " << ii << "  histo[ii]: " << histo[ii] << endl;

	    checked[ii] = true;
	    continue;
	  }
          break;
        }
        for(int ii = peak1_id - 1; ii >= 0; ii --) {
          if(histo[ii] <= histo[ii + 1]) {
//if(n == 0 && m == 60) cerr << " -----   ii: " << ii << "  histo[ii]: " << histo[ii] << endl;
	    checked[ii] = true;
	    continue;
	  }
          break;
        }

        int peak2 = 0;
        int peak2_id = 0;
        for(int ii = 0; ii < items; ii ++) {
          if(checked[ii]) continue;
          if(histo[ii] > peak2) {
	    peak2 = histo[ii];
	    peak2_id = ii;
          }
        }
        checked[peak2_id] = true;
        
        for(int ii = peak2_id + 1; ii < items; ii ++) {
          if(checked[ii]) continue;
          if(histo[ii] <= histo[ii - 1]) {
	    checked[ii] = true;
	    continue;
	  }
          break;
        }
        for(int ii = peak2_id - 1; ii >= 0; ii --) {
          if(checked[ii]) continue;
          if(histo[ii] <= histo[ii + 1]) {
	    checked[ii] = true;
	    continue;
	  }
          break;
        }
        
/*
        if(n == 0 && m == 60) {
	  FILE *fp = fopen("0_60.dat", "w");
          for(int ii = 0; ii < items; ii ++) {
	    fprintf(fp, "%d  %d  %d  %lf\n", ii, histo[ii], (int)checked[ii], minv + delta * ii);
	  }
          fclose(fp);
          cerr << "n: " << n << "  m: " << m << "  peak1: " << peak1 << "  peak1_id: " << peak1_id << "  peak2: " << peak2 << "  peak2_id: " << peak2_id << endl;
	}
        if(n == 0 && m == 73) {
	  FILE *fp = fopen("0_73.dat", "w");
          for(int ii = 0; ii < items; ii ++) {
	    fprintf(fp, "%d  %d  %d  %lf\n", ii, histo[ii], (int)checked[ii] , minv + delta * ii);
	  }
          fclose(fp);
          cerr << "n: " << n << "  m: " << m << "  peak1: " << peak1 << "  peak1_id: " << peak1_id << "  peak2: " << peak2 << "  peak2_id: " << peak2_id << endl;
         // exit(0);
	}
*/


        delete[] checked;
        delete[] histo;
          
        if(count > 1) {
          if(change_type == 0) {
	    double ratio = (minv + delta * peak1_id)/(minv + delta * peak2_id);
	    if(ratio > change_th) {
              for(int i = 0; i < box_size; i ++) {
                if(start_line + i >= nr_lines) continue;
	        for(int j = 0; j < box_size; j++) {
	    	  if(start_pixel + j >= nr_pixels) continue;
		  change_data[start_line + i][start_pixel + j] ++;
		}
	      }
	    }
	  }
          else if(change_type == 1) {
            double diff = delta * (peak1_id - peak2_id);
	    if(diff > change_th) {
              for(int i = 0; i < box_size; i ++) {
                if(start_line + i >= nr_lines) continue;
	        for(int j = 0; j < box_size; j++) {
	    	  if(start_pixel + j >= nr_pixels) continue;
		  change_data[start_line + i][start_pixel + j] ++;
		}
	      }
	    }
	  }
	}

      }
      delete work_patch;
    }

    box_size ++;
  }

  max_change = 0;
  for(int i = 0; i < nr_lines; i++) {
    for(int j = 0; j < nr_pixels; j ++) {
      if(change_data[i][j] > max_change) max_change = change_data[i][j];
    }
  }
  cerr << "ChangeDetector::max_change: " << max_change << endl;
}


