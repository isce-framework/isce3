// Copyright (c) 2017-, California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged.
// All rights reserved.
// 
// Author(s):
// 
//  ----------------------------------------------------------------------------
//  Author:  Xiaoqing Wu
// 

#include "CannyEdgeDetector.h"
#include "sort.h"

void CannyEdgeDetector::basic_init()
{
  nr_lines = 0;
  nr_pixels = 0;
  no_data = 0;
  gw_size = 0;
  gw_sigma = 0;
  low_th = 0;
  high_th = 0;

  edge = NULL;
  //data_patch = NULL;
  //edge_patch = NULL;
}

CannyEdgeDetector::~CannyEdgeDetector()
{ 
  for(int i = 0; i < nr_lines; i++) {
    delete[] edge[i]; 
  }
  delete edge;
}

CannyEdgeDetector::CannyEdgeDetector(int nr_lines_, int nr_pixels_, float no_data_, float **data, 
		                     float low, float high, int gw, double gw_sigma_)
{
  basic_init();
  nr_lines = nr_lines_;
  nr_pixels = nr_pixels_;
  no_data = no_data_;
  low_th = low;
  high_th = high;
  gw_size = gw;
  gw_sigma = gw_sigma_;


  edge = new float *[nr_lines];
  for(int i = 0; i < nr_lines; i++) {
    edge[i] = new float[nr_pixels];
  }
  calculate(data);
}


void CannyEdgeDetector::calculate(float **data)
{

  cerr << "  starting CannyEdgeDetector::calculate() ...... \n";

// 1. do Gaussian filtering 

  int gw = gw_size/2;

  DataPatch<float> *tmp_edge_patch = new DataPatch<float>(nr_pixels, nr_lines);
  float **tmp_edge = tmp_edge_patch->get_data_lines_ptr();
  for(int i = 0; i < nr_lines; i ++) {
    for(int j = 0; j < nr_pixels; j ++) {
      tmp_edge[i][j] = 0;
    }
  }

if(gw_sigma > 0.00001) {
  double e = 2.71828182846;

  DataPatch<float> *gaussian_patch = new DataPatch<float>(gw_size, gw_size);
  float **gaussian = gaussian_patch->get_data_lines_ptr();


  double count = 0;
  for(int i = 0; i < gw_size; i++) {
    double x = (i - gw);
    for(int j = 0; j < gw_size; j++) {
      double y = (j - gw);
      double t = -(x*x + y*y)/(2.0 * gw_sigma * gw_sigma);
      gaussian[i][j] = pow(e, t);
      count += gaussian[i][j];
    }
  }
  
  for(int i = 0; i < gw_size; i++) {
    for(int j = 0; j < gw_size; j++) {
      gaussian[i][j] /= count;
    }
  }


 # pragma omp parallel for
  for(int i = 0; i < nr_lines; i ++) {
    for(int j = 0; j < nr_pixels; j ++) {
      double sum = 0;
      double count = 0;
      for(int ii = i - gw; ii <= i + gw; ii++) {
        if(ii < 0 || ii >= nr_lines) continue;
	for(int jj = j - gw; jj <= j + gw; jj ++) {
	  if(jj < 0 || jj >= nr_pixels) continue;
	  if(data[ii][jj] == no_data) continue;
	  int k1 = ii - i + gw;
	  int k2 = jj - j + gw;
	  sum += data[ii][jj] * gaussian[k1][k2];
   	  count += gaussian[k1][k2];
	}
      }
      if(count > 0) {
        sum /= count;
        tmp_edge[i][j] = sum;
      }
      else {
	tmp_edge[i][j] = 0;
      }
    }
  }
  delete gaussian_patch;
}
else {  // do medium filtering 

  for(int i = 0; i < nr_lines; i ++) {
    for(int j = 0; j < nr_pixels; j ++) {
      tmp_edge[i][j] = data[i][j];
    }
  }
/*
  int win = gw;
  int win2 = 2*win + 1;

cerr << "win: " << win << "  win2: " << win2 << endl;

    int *indexes = new int[win2];
    double *work = new double[win2];
// # pragma omp parallel for
  for(int i = 0; i < nr_lines; i++) {
 
cerr << "i: " << i << endl;
//    int *indexes = new int[win2];
//    double *work = new double[win2];

    for(int j = 0; j < nr_pixels; j++) {
//cerr << "j: " << j << endl;
      int count = 0;
      for(int ii = i - win; ii <= i + win; ii++) {
        if(ii < 0 || ii >= nr_lines) continue;
        for(int jj = j - win; jj <= j + win; jj ++) {
          if(jj < 0 || jj >= nr_pixels) continue;
          if(data[ii][jj] == no_data) continue;
	  indexes[count] = count;
          work[count] = data[ii][jj];
          count ++;
	}
      }
//cerr << "count: " << count << endl;
      if(count == 0) tmp_edge[i][j] = no_data;
      else if(count <= 4) {
        double sum = 0;
        for(int k = 0; k < count; k++) {
	  sum += work[k];
        }
	tmp_edge[i][j] = sum / (double) count;
      }
      else {
        heapSort (count, work, indexes);
        tmp_edge[i][j] = work[indexes[count/2]];
      }
    }
//    delete[] indexes;
//    delete[] work;	  
  }
    delete[] indexes;
    delete[] work;
*/	  
}

  cerr << "end of gaussian filtering !!!!!! \n";

  if(0) {
    FILE *fp = fopen("filtered.dat", "w");
    for(int i = 0; i < nr_lines; i ++) {
      fwrite(tmp_edge[i], 4, nr_pixels, fp);
    }
    fclose(fp);    
  }
// 2.  calculate gradient or edge strength and determine the direction of the edge

  DataPatch<unsigned char> *dir_patch = new DataPatch<unsigned char>(nr_pixels, nr_lines); 
  unsigned char **dir = dir_patch->get_data_lines_ptr();

  double eighth = PI/8.0; 

  for(int i = 2; i < nr_lines - 1; i ++) {
    for(int j = 2; j < nr_pixels - 1; j ++) {
      double gx = tmp_edge[i - 2][j] - tmp_edge[i - 2][j - 1] + 
                  2.0*tmp_edge[i - 1][j] - 2.0*tmp_edge[i - 1][j - 1] +
                  2.0*tmp_edge[i][j] - 2.0*tmp_edge[i][j - 1] +
                  tmp_edge[i + 1][j] - tmp_edge[i + 1][j - 1] ;
      double gy = tmp_edge[i][j - 2] - tmp_edge[i - 1][j - 2] + 
                  2.0*tmp_edge[i][j - 1] - 2.0*tmp_edge[i - 1][j - 1] +
                  2.0*tmp_edge[i][j] - 2.0*tmp_edge[i - 1][j] +
                  tmp_edge[i][j + 1] - tmp_edge[i - 1][j + 1] ;
      
      edge[i][j] = fabs(gx) + fabs(gy);
      double theta = atan2(gy, gx);
      if(theta < 0) theta += PI;

      if(theta <= eighth || theta >= PI - eighth) dir[i][j] = 1;
      else if(fabs(theta - PI/4.0) <= eighth) dir[i][j] = 2;
      else if(fabs(theta - PI/2.0) <= eighth) dir[i][j] = 3;
      else if(fabs(theta - PI*0.75) <= eighth) dir[i][j] = 4;
      else dir[i][j] = 0;
    }
  }

  cerr << "End of edge streng calculation !!!!!! \n";

// 3. non-maxima suppression 


  for(int i = 0; i < nr_lines; i ++) {
    for(int j = 0; j < nr_pixels; j ++) {
      tmp_edge[i][j] = edge[i][j];
    }
  }

  for(int i = 2; i < nr_lines - 1; i ++) {
    for(int j = 2; j < nr_pixels - 1; j ++) {
      if(dir[i][j] == 1) {
	if(tmp_edge[i][j-1] >= tmp_edge[i][j] || tmp_edge[i][j+1] >= tmp_edge[i][j]) {
	  edge[i][j] = 0;
	}
      }
      else if(dir[i][j] == 2) {
	if(tmp_edge[i-1][j-1] >= tmp_edge[i][j] || tmp_edge[i+1][j+1] >= tmp_edge[i][j]) {
	  edge[i][j] = 0;
	}
      }
      else if(dir[i][j] == 3) {
	if(tmp_edge[i-1][j] >= tmp_edge[i][j] || tmp_edge[i+1][j] >= tmp_edge[i][j]) {
	  edge[i][j] = 0;
	}
      }
      else if(dir[i][j] == 4) {
	if(tmp_edge[i+1][j-1] >= tmp_edge[i][j] || tmp_edge[i-1][j+1] >= tmp_edge[i][j]) {
	  edge[i][j] = 0;
	}
      }
    }
  }


/*
  for(int i = 1; i < nr_lines - 1; i ++) {
    for(int j = 1; j < nr_pixels - 1; j ++) {
      double gx = tmp_edge[i - 1][j + 1] - tmp_edge[i - 1][j - 1] + 
                  2.0*tmp_edge[i][j + 1] - 2.0*tmp_edge[i][j - 1] +
                  tmp_edge[i + 1][j + 1] - tmp_edge[i + 1][j - 1] ;
      double gy = tmp_edge[i + 1][j - 1] - tmp_edge[i - 1][j - 1] + 
                  2.0*tmp_edge[i + 1][j] - 2.0*tmp_edge[i - 1][j] +
                  tmp_edge[i + 1][j + 1] - tmp_edge[i - 1][j + 1] ;
      
      edge[i][j] = fabs(gx) + fabs(gy);
      double theta = atan2(gy, gx);
      if(theta < 0) theta += PI;

      if(theta <= eighth || theta >= PI - eighth) dir[i][j] = 1;
      else if(fabs(theta - PI/4.0) <= eighth) dir[i][j] = 2;
      else if(fabs(theta - PI/2.0) <= eighth) dir[i][j] = 3;
      else if(fabs(theta - PI*0.75) <= eighth) dir[i][j] = 4;
      else dir[i][j] = 0;
    }
  }

  cerr << "End of edge streng calculation !!!!!! \n";

  if(0){
    FILE *fp = fopen("edge0.dat", "w");
    for(int i = 0; i < nr_lines; i ++) {
      fwrite(edge[i], 4, nr_pixels, fp);
    }
    fclose(fp);    
  }

// 3. non-maxima suppression 


  for(int i = 0; i < nr_lines; i ++) {
    for(int j = 0; j < nr_pixels; j ++) {
      tmp_edge[i][j] = edge[i][j];
    }
  }

  for(int i = 1; i < nr_lines - 1; i ++) {
    for(int j = 1; j < nr_pixels - 1; j ++) {
      if(dir[i][j] == 1) {
	if(tmp_edge[i][j-1] >= tmp_edge[i][j] || tmp_edge[i][j+1] >= tmp_edge[i][j]) {
	  edge[i][j] = 0;
	}
      }
      else if(dir[i][j] == 2) {
	if(tmp_edge[i-1][j-1] >= tmp_edge[i][j] || tmp_edge[i+1][j+1] >= tmp_edge[i][j]) {
	  edge[i][j] = 0;
	}
      }
      else if(dir[i][j] == 3) {
	if(tmp_edge[i-1][j] >= tmp_edge[i][j] || tmp_edge[i+1][j] >= tmp_edge[i][j]) {
	  edge[i][j] = 0;
	}
      }
      else if(dir[i][j] == 4) {
	if(tmp_edge[i+1][j-1] >= tmp_edge[i][j] || tmp_edge[i-1][j+1] >= tmp_edge[i][j]) {
	  edge[i][j] = 0;
	}
      }
    }
  }
*/
  delete tmp_edge_patch;
  delete dir_patch;


  if(0){
    FILE *fp = fopen("T1.dat", "w");
    for(int i = 0; i < nr_lines; i ++) {
      for(int j = 0; j < nr_pixels; j ++) {
	float x = edge[i][j];
	if(x <= low_th) x = 0;
	fwrite(&x, 4, 1, fp);
      }
    }
    fclose(fp);    
  }

  if(0){
    FILE *fp = fopen("T2.dat", "w");
    for(int i = 0; i < nr_lines; i ++) {
      for(int j = 0; j < nr_pixels; j ++) {
	float x = edge[i][j];
	if(x <= high_th) x = 0;
	fwrite(&x, 4, 1, fp);
      }
    }
    fclose(fp);    
  }

// 4. thresholding with 2 thresholds

  double T1 = low_th;
  double T2 = high_th;
	

  DataPatch<char> *visit_patch = new DataPatch<char>(nr_pixels, nr_lines);
  char **visit = visit_patch->get_data_lines_ptr();
  
  char not_visited = 0;
  char visited = 1;

  for(int line = 0; line < nr_lines; line ++) {
    for(int pixel = 0; pixel < nr_pixels; pixel ++) {
      visit[line][pixel] = not_visited;
    }
  }
  
  queue<Point> workq;

  for(int line = 1; line < nr_lines - 1; line ++) {
    for(int pixel = 1; pixel < nr_pixels - 1; pixel ++) {
      if(edge[line][pixel] > T2) workq.push(Point(pixel, line));
    }
  }

  while( !workq.empty() ) {
    Point point = workq.front();
    workq.pop();
    int line  = point.get_Y();
    int pixel = point.get_X();
    visit[line][pixel] = visited;

    int line_plus = line + 1;
    int line_minus = line - 1;
    int pixel_plus = pixel + 1;
    int pixel_minus = pixel - 1;
      
      if(line > 0) {              // facing up ......
	if(edge[line_minus][pixel] > T1 && visit[line_minus][pixel] == not_visited) {
	  workq.push(Point(pixel, line_minus));
	  visit[line_minus][pixel] = visited;
	}	
      }
      if(line < nr_lines - 1) {   // facing down ...... 
	if(edge[line_plus][pixel] > T1 && visit[line_plus][pixel] == not_visited) {
	  workq.push(Point(pixel, line_plus));
	  visit[line_plus][pixel] = visited;
	}
      }	
      if(pixel > 0) {             // facing left ......
	if(edge[line][pixel_minus] > T1 && visit[line][pixel_minus] == not_visited) {
	  workq.push(Point(pixel_minus, line));
	  visit[line][pixel_minus] = visited;
	}
      }
      if(pixel < nr_pixels - 1) {// facing right ......
	if(edge[line][pixel_plus] > T1 && visit[line][pixel_plus] == not_visited) {
	  workq.push(Point(pixel_plus, line));
	  visit[line][pixel_plus] = visited;
	}
      }
    
  }  

  
  for(int line = 1; line < nr_lines; line ++) {
    for(int pixel = 1; pixel < nr_pixels; pixel ++) {
      if(edge[line][pixel] > T2) edge[line][pixel] = T2;
      else if(edge[line][pixel] > T1 && visit[line][pixel] == visited) ;
      else edge[line][pixel] = 0;
    }
  }
/*
  if(1){
    FILE *fp = fopen("edge2.dat", "w");
    for(int i = 0; i < nr_lines; i ++) {
      fwrite(edge[i], 4, nr_pixels, fp);
    }
    fclose(fp);    
  }
*/
}

  

