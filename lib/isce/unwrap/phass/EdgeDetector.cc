// Copyright (c) 2017-, California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged.
// All rights reserved.
// 
// Author(s):
// 
//  ======================================================================
// 
//  FILENAME: EdgeDetector.cc
// 
//  CREATED BY: Xiaoqing WU
// 
//  ======================================================================
#include "EdgeDetector.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

using namespace std;


void detect_edge(int nr_lines, int nr_pixels, float **data, unsigned char **edge_data,
		 int window_length, double C_min, double R_edge, double R_line)
{
  unsigned char Zero = (unsigned char)0;
  unsigned char One  = (unsigned char)1;
  
  for(int line = 0; line < nr_lines; line ++) {
    for(int pixel = 0; pixel < nr_pixels; pixel ++) {
      edge_data[line][pixel] = Zero;
    }
  }

  cerr << "C_min: " << C_min << "  R_edge: " << R_edge << "  R_line: " << R_line << endl;

  int win = window_length/2;
  int start_line = win;
  int stop_line = nr_lines - win;
  int start_pixel = win;
  int stop_pixel = nr_pixels - win;

  cerr << "window_length: " << window_length << "  win: " << win << endl;

 # pragma omp parallel for
  for(int line = start_line; line < stop_line; line ++) {


    for(int pixel = start_pixel; pixel < stop_pixel; pixel ++) {

      // calculate the Coefficient of variation Ca first ......
      double mean = 0;
      for(int ii = line - win; ii <= line + win; ii++) {
        for(int jj = pixel - win; jj <= pixel + win; jj ++) {
	  mean += data[ii][jj];
	}
      }
      mean /= (window_length * window_length);

      double deviation = 0;
      for(int ii = line - win; ii <= line + win; ii++) {
        for(int jj = pixel - win; jj <= pixel + win; jj ++) {
	  double x = data[ii][jj] - mean;
 	  deviation += x*x;
	}
      }
//     if(line > 5995 ) cerr << "  line: " << line << "  pixel: " << pixel << "  mean: " << mean << "  deviation: " << deviation << endl;
      if(mean <= 1.0e-20 && deviation <= 1.0e-20) continue;
      deviation /= (double)(window_length * window_length - 1);
      

      double Ca = sqrt(deviation)/mean;



      if(Ca < C_min) continue;      // Homogenous area with no edge      

// Check Edge structures ......
      // horizontal edge ......
      double x1 = 0;
      double x2 = 0;
      for(int ii = line - win; ii < line; ii++) {
	int ii2 = ii + win + 1;
        for(int jj = pixel - win; jj <= pixel + win; jj ++) {
	  x1 += data[ii][jj];
	  x2 += data[ii2][jj];
	}
      }
      if(x1 == 0 || x2 == 0) continue;
      double R1 = x1/x2;
      if(R1 > 1.0) R1 = 1.0/R1;

      // Vertical edge ......
      x1 = 0;
      x2 = 0;
      for(int ii = line - win; ii <= line + win; ii++) {
        for(int jj = pixel - win; jj < pixel; jj ++) {
	  int jj2 = jj + win + 1;
	  x1 += data[ii][jj];
	  x2 += data[ii][jj2];
	}
      }
      if(x1 == 0 || x2 == 0) continue;
      double R2 = x1/x2;
      if(R2 > 1.0) R2 = 1.0/R2;

      // Diagonal left high ......
      x1 = 0;
      x2 = 0;
      for(int ii = line - win; ii <= line + win; ii++) {
        int d_ii = ii - line;
        for(int jj = pixel - win; jj <= pixel + win; jj ++) {
	  int d_jj = jj - pixel;
          if(d_jj > d_ii) x1 += data[ii][jj];
          if(d_jj < d_ii) x2 += data[ii][jj];
	}
      }
      if(x1 == 0 || x2 == 0) continue;
      double R3 = x1/x2;
      if(R3 > 1.0) R3 = 1.0/R3;

      // Diagonal right high ......
      x1 = 0;
      x2 = 0;
      for(int ii = line - win; ii <= line + win; ii++) {
        int d_ii = ii - line;
        for(int jj = pixel - win; jj <= pixel + win; jj ++) {
	  int d_jj = -jj + pixel;
          if(d_jj > d_ii) x1 += data[ii][jj];
          if(d_jj < d_ii) x2 += data[ii][jj];
	}
      }
      if(x1 == 0 || x2 == 0) continue;
      double R4 = x1/x2;
      if(R4 > 1.0) R4 = 1.0/R4;

      double R = 1.0;
      if(R1 < R) R = R1;
      if(R2 < R) R = R2;
      if(R3 < R) R = R3;
      if(R4 < R) R = R4;

      if(R < R_edge) edge_data[line][pixel] = (unsigned char)(255 * (R_edge - R)/R_edge);


// Check line structures ......

      // horizontal line ......
      x1 = 0;
      for(int ii = line - win; ii < line; ii++) {
	int ii2 = ii + win + 1;
        for(int jj = pixel - win; jj <= pixel + win; jj ++) {
	  x1 += data[ii][jj] + data[ii2][jj];
	}
      }
      x2 = 0;
      for(int jj = pixel - win; jj <= pixel + win; jj ++) x2 += data[line][jj];
      if(x1 == 0 || x2 == 0) continue;
      R1 = x1/x2/2.0/(double)win;
      if(R1 > 1.0) R1 = 1.0/R1;

      // Vertical edge ......
      x1 = 0;
      x2 = 0;
      for(int ii = line - win; ii <= line + win; ii++) {
        for(int jj = pixel - win; jj < pixel; jj ++) {
	  int jj2 = jj + win + 1;
	  x1 += data[ii][jj] + data[ii][jj2];
	}
      }
      x2 = 0;
      for(int ii = line - win; ii <= line + win; ii++) x2 += data[ii][pixel];
      if(x1 == 0 || x2 == 0) continue;
      R2 = x1/x2/2.0/(double)win;
      if(R2 > 1.0) R2 = 1.0/R2;

      // Diagonal left high ......
      x1 = 0;
      for(int ii = line - win; ii <= line + win; ii++) {
        int d_ii = ii - line;
        for(int jj = pixel - win; jj <= pixel + win; jj ++) {
	  int d_jj = jj - pixel;
          if(d_jj > d_ii) x1 += data[ii][jj];
          if(d_jj < d_ii) x1 += data[ii][jj];
	}
      }
      x2 = 0;
      for(int ii = line - win; ii <= line + win; ii++) {
        int d_ii = ii - line;
	x2 += data[ii][pixel + d_ii];
      }
      if(x1 == 0 || x2 == 0) continue;
      R3 = x1/x2/2.0/(double)win;
      if(R3 > 1.0) R3 = 1.0/R3;

      // Diagonal right high ......
      x1 = 0;
      for(int ii = line - win; ii <= line + win; ii++) {
        int d_ii = ii - line;
        for(int jj = pixel - win; jj <= pixel + win; jj ++) {
	  int d_jj = -jj + pixel;
          if(d_jj > d_ii) x1 += data[ii][jj];
          if(d_jj < d_ii) x1 += data[ii][jj];
	}
      }
      x2 = 0;
      for(int ii = line - win; ii <= line + win; ii++) {
        int d_ii = ii - line;
	x2 += data[ii][pixel - d_ii];
      }

      if(x1 == 0 || x2 == 0) continue;
      R4 = x1/x2/2.0/(double)win;
      if(R4 > 1.0) R4 = 1.0/R4;

      R = 1.0;
      if(R1 < R) R = R1;
      if(R2 < R) R = R2;
      if(R3 < R) R = R3;
      if(R4 < R) R = R4;

      if(R < R_line) edge_data[line][pixel] = (unsigned char)(255 * (R_line - R)/R_line);

    }
  }


  for(int line = 0; line < win; line ++) {
    for(int pixel = 0; pixel < nr_pixels; pixel ++) {
      unsigned char maxValue = 0;
      for(int ii = line - win; ii <= line + win; ii ++) {
        if(ii < 0) continue;
	if(edge_data[ii][pixel] > maxValue) maxValue = edge_data[ii][pixel];
      }
      edge_data[line][pixel] = maxValue;
    }
  }


  for(int line = nr_lines - win; line < nr_lines; line ++) {
    for(int pixel = 0; pixel < nr_pixels; pixel ++) {
      unsigned char maxValue = 0;
      for(int ii = line - win; ii <= line + win; ii ++) {
        if(ii >= nr_lines) continue;
	if(edge_data[ii][pixel] > maxValue) maxValue = edge_data[ii][pixel];
      }
      edge_data[line][pixel] = maxValue;
    }
  }


  for(int pixel = 0; pixel < win; pixel ++) {  
    for(int line = 0; line < nr_lines; line ++) {
      unsigned char maxValue = 0;
      for(int jj = pixel - win; jj <= pixel + win; jj++) {
	if(jj < 0) continue;
	if(edge_data[line][jj] > maxValue) maxValue = edge_data[line][jj];
      }
      edge_data[line][pixel] = maxValue;
    }
  }


  for(int pixel = nr_pixels - win; pixel < nr_pixels; pixel ++) {  
    for(int line = 0; line < nr_lines; line ++) {
      unsigned char maxValue = 0;
      for(int jj = pixel - win; jj <= pixel + win; jj++) {
	if(jj >= nr_pixels) continue;
	if(edge_data[line][jj] > maxValue) maxValue = edge_data[line][jj];
      }
      edge_data[line][pixel] = maxValue;
    }
  }
	
}


void detect_edge(int nr_lines, int nr_pixels, float **data, 
		 unsigned char **horizontal_edge_data, unsigned char **vertical_edge_data,
		 int window_length, double coefficient_variance_min, double max_edge_ratio)
{
  unsigned char Zero = (unsigned char)0;
  unsigned char One  = (unsigned char)1;
  double Ca_min = coefficient_variance_min;
  
  for(int line = 0; line < nr_lines; line ++) {
    for(int pixel = 0; pixel < nr_pixels; pixel ++) {
      horizontal_edge_data[line][pixel] = Zero;
      vertical_edge_data[line][pixel] = Zero;
    }
  }


  DataPatch<float> *mean_patch = new DataPatch<float>(nr_pixels, nr_lines);
  float **mean_data = mean_patch->get_data_lines_ptr();
  DataPatch<float> *ca_patch = new DataPatch<float>(nr_pixels, nr_lines);
  float **ca_data = ca_patch->get_data_lines_ptr();

  int win = window_length/2;
  int start_line = 0;  // win;
  int stop_line = nr_lines;   //nr_lines - win;
  int start_pixel = 0; // win;
  int stop_pixel = nr_pixels; // nr_pixels - win;

  double Cu2 = coefficient_variance_min * coefficient_variance_min;
  cerr << "window_length: " << window_length << "  win: " << win << endl;

# pragma omp parallel for
  for(int line = start_line; line < stop_line; line ++) {
    for(int pixel = start_pixel; pixel < stop_pixel; pixel ++) {


      // calculate the Coefficient of variation Ca first ......
      double mean = 0;
      int count = 0;
      for(int ii = line - win; ii <= line + win; ii++) {
        if(ii < 0 || ii >= nr_lines) continue;
        for(int jj = pixel - win; jj <= pixel + win; jj ++) {
	  if(jj < 0 || jj >= nr_pixels) continue;
	  mean += data[ii][jj];
	  count ++;
	}
      }
      if(count < window_length) continue;
      mean /= (double)count;

      double deviation = 0;
      for(int ii = line - win; ii <= line + win; ii++) {
        if(ii < 0 || ii >= nr_lines) continue;
        for(int jj = pixel - win; jj <= pixel + win; jj ++) {
	  if(jj < 0 || jj >= nr_pixels) continue;
	  double x = data[ii][jj] - mean;
 	  deviation += x*x;
	}
      }

      if(mean <= 1.0e-20 && deviation <= 1.0e-20) continue;
      deviation /= (double)(window_length * window_length - 1);
      double Ca = sqrt(deviation)/mean;


//      if(Ca < coefficient_variance_min) continue;      // Homogenous area with no edge  
      
      double weight = (1.0 - Cu2/Ca/Ca)/(1 + Cu2);
      ca_data[line][pixel] = weight;
      mean_data[line][pixel] = mean;
      
    }
  }

# pragma omp parallel for
  for(int line = start_line; line < stop_line; line ++) {
    for(int pixel = start_pixel; pixel < stop_pixel; pixel ++) {
      double Ca = ca_data[line][pixel];
      if(Ca < coefficient_variance_min) continue;
      double weight = (1.0 - Cu2/Ca/Ca)/(1 + Cu2);;
      mean_data[line][pixel] =  mean_data[line][pixel] * weight + data[line][pixel] * (1.0 - weight);
    }
  }
  delete ca_patch;

# pragma omp parallel for
  for(int line = start_line; line < stop_line; line ++) {
    for(int pixel = start_pixel; pixel < stop_pixel; pixel ++) {
// Check Edge structures ......
      // horizontal edge ......
      int ii = line;
      int ii2 = ii + 1;
      if(ii2 < nr_lines) {
        double x1 = 0;
        double x2 = 0;
        for(int jj = pixel - win; jj <= pixel + win; jj ++) {
          if(jj < 0 || jj >= nr_pixels) continue;
	  x1 += mean_data[ii][jj];
	  x2 += mean_data[ii2][jj];
//	  x1 += data[ii][jj];
//	  x2 += data[ii2][jj];
	}
        if(x1 == 0 || x2 == 0) continue;
        double ratio = x1/x2;
        if(ratio > 1.0) ratio = 1.0/ratio;
        ratio = (max_edge_ratio - ratio)/max_edge_ratio;
        if(ratio > 0) {
          horizontal_edge_data[line][pixel] = (unsigned char)(255*ratio);
	}
      }

      // Vertical edge ......
      int jj = pixel;
      int jj2 = pixel + 1;
      if(jj2 < nr_pixels) {
        double x1 = 0;
        double x2 = 0;
        for(int ii = line - win; ii <= line + win; ii++) {
          if(ii < 0 || ii >= nr_lines) continue;
	  x1 += mean_data[ii][jj];
	  x2 += mean_data[ii][jj2];
//	  x1 += data[ii][jj];
//	  x2 += data[ii][jj2];
	}
        if(x1 == 0 || x2 == 0) continue;
        double ratio = x1/x2;
        if(ratio > 1.0) ratio = 1.0/ratio;
        ratio = (max_edge_ratio - ratio)/max_edge_ratio;
        if(ratio > 0) {
          vertical_edge_data[line][pixel] = (unsigned char)(255*ratio);
	}

      }

    }
  }

  delete mean_patch;
}
