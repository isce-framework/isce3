// Copyright (c) 2017-, California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged.
// All rights reserved.
// 
// Author(s):
// 
//  ======================================================================
// 
//  FILENAME: PhassUnwrapper.cc
//   
//  CREATED BY: Xiaoqing WU
// 
//  ======================================================================
#include "CannyEdgeDetector.h"
#include "PhassUnwrapper.h"
#include "ASSP.h"
#include "DataPatch.h"
#include "Point.h"
#include <vector>  
#include <set>   
#include <list>   
#include <queue>  
#include <omp.h> 

void phass_unwrap(int nr_lines, int nr_pixels, float **phase_data, float **corr_data, float **power_data, int **region_map,
		  double corr_th, double good_corr, int min_pixels_per_region)
{

  cerr << "phass_unwrap input parameters:: ......\n";
  cerr << "     nr_lines: " << nr_lines << endl;
  cerr << "     nr_pixels: " << nr_pixels << endl;
  cerr << "     corr_th: " << corr_th << endl;
  cerr << "     good_corr: " << good_corr << endl;
  cerr << "     min_pixels_per_region: " << min_pixels_per_region << endl;
/*
  {
    fcomplex *int_data = new fcomplex[nr_pixels];
    FILE *fp = fopen("int_data.dat", "w");
    for(int line = 0; line < nr_lines; line ++) {
      for(int pixel = 0; pixel < nr_pixels; pixel ++) {
        int_data[pixel] = corr_data[line][pixel] * fcomplex(cos(phase_data[line][pixel]), sin(phase_data[line][pixel]));
      }
      fwrite(int_data, sizeof(fcomplex), nr_pixels, fp);
    }
    delete int_data;
    fclose(fp);
  }
*/
/*
  {
    FILE *fp = fopen("phase_data.dat", "w");
    FILE *fp2 = fopen("corr_data.dat", "w");
    FILE *fp3 = fopen("power_data.dat", "w");
    for(int line = 0; line < nr_lines; line ++) {
      fwrite(phase_data[line], sizeof(float), nr_pixels, fp);
      fwrite(corr_data[line], sizeof(float), nr_pixels, fp2);
      fwrite(power_data[line], sizeof(float), nr_pixels, fp3);
    }
    fclose(fp);
    fclose(fp2);
    fclose(fp3);
//    exit(0);
  }
*/

  double small = 0.000000001;
  double phase_diff_th = 1.0; // radians

  // default to square the corr_data ......
  for(int line = 0; line < nr_lines; line ++) {
    for(int pixel = 0; pixel < nr_pixels; pixel ++) {
      corr_data[line][pixel] *= corr_data[line][pixel];
    }
  }
  corr_th *= corr_th;
  good_corr *= good_corr;

// (1) make node data .................

  int nrows = nr_lines + 1;
  int ncols = nr_pixels + 1;

  DataPatch<Node> *node_patch = new DataPatch<Node>(ncols, nrows);
  Node **node_data = node_patch->get_data_lines_ptr();
  for(int row = 0; row < nrows; row++) {
    for(int col = 0; col < ncols; col ++) {
      node_data[row][col].supply = 0;
      node_data[row][col].rc = 0;	  
      node_data[row][col].dc = 0;
    }
  }
  
  double pi = PI;
  double two_pi = 2.0 * PI;
  float *phases = new float[5];
  for(int line=1; line<nr_lines; line++) {
    for(int pixel=1; pixel<nr_pixels; pixel++) {
      phases[0] = phase_data[line-1][pixel-1];
      phases[1] = phase_data[line][pixel-1];
      phases[2] = phase_data[line][pixel];
      phases[3] = phase_data[line-1][pixel];
      phases[4] = phases[0];
      char flag = 0;
      for(int k=0; k< 4; k++) {
	double x = phases[k+1] - phases[k];
	if(x < -pi) flag += -1;
	if(x >= pi)  flag +=  1;
      }
      node_data[line][pixel].supply = flag;
    }
  }
  delete[] phases;

  double x, y;
  int mask_th = good_corr * cost_scale;
  for(int line = 0; line < nrows; line++) {
    for(int pixel = 0; pixel < ncols; pixel ++) {
      if(line == 0) {        // For the first row ......
	if(pixel > 0 && pixel < ncols - 1) {
	  x = min(corr_data[line][pixel - 1], corr_data[line][pixel]);
	  node_data[line][pixel].dc = (uchar)(x * cost_scale);
	}
      }
      else if(line < nrows - 1) {    // For middle rows ......
	if(pixel == 0) {
	  y = min(corr_data[line - 1][pixel], corr_data[line][pixel]);
	  node_data[line][pixel].rc = (uchar)(y * cost_scale);
	}
	else {
	  y = min(corr_data[line - 1][pixel], corr_data[line][pixel]);
	  node_data[line][pixel].rc = (uchar)(y * cost_scale);
	  x = min(corr_data[line][pixel - 1], corr_data[line][pixel]);
	  node_data[line][pixel].dc = (uchar)(x * cost_scale);
	}
      }
      if(node_data[line][pixel].dc > mask_th) node_data[line][pixel].dc = 255;
      if(node_data[line][pixel].rc > mask_th) node_data[line][pixel].rc = 255;	  
    }
  }

  if(power_data != NULL) {   // set cuts based on the detected edges
    float low = 4.0;
    float high = 12.0;
    float mid = 8.0;
    int gw = 7;
    double gws = 1.0;
    for(int i = 0; i < nr_lines; i++) {
      for(int j = 0; j < nr_pixels; j++) {
        power_data[i][j] = 10.0*log10(power_data[i][j]+1.0e-20);  // for power inputs
      }
    }
    CannyEdgeDetector *edgeDetector = new CannyEdgeDetector(nr_lines, nr_pixels, -200.0, power_data, low, high, gw, gws);
    float **edge = edgeDetector->get_edge();
    for(int i = 1; i < nr_lines - 1; i ++) {
      for(int j = 1; j < nr_pixels - 1; j ++) {
	if(edge[i][j] >= mid) {
	  if(edge[i + 1][j] >= mid) node_data[i][j].dc = 0;
	  if(edge[i][j + 1] >= mid) node_data[i][j].rc = 0;
          if(edge[i - 1][j] >= mid) node_data[i - 1][j].dc = 0;
	  if(edge[i][j - 1] >= mid) node_data[i][j - 1].rc = 0;
	}
      }
    }
    delete edgeDetector;
  }


  double max_dph = phase_diff_th; //1.0; // PI/2.0;
  double dx = 0;
  for(int line = 0; line < nrows; line++) {
    for(int pixel = 0; pixel < ncols; pixel ++) {
      if(line == 0) {        // For the first row ......
	if(pixel > 0 && pixel < ncols - 1) {
	  if(corr_data[line][pixel] > small && corr_data[line][pixel - 1] > small) {
	    dx = phase_data[line][pixel] - phase_data[line][pixel - 1];
	    dx = fabs(dx);
            if(dx > PI) dx = two_pi - dx;
	    if(dx >= max_dph) node_data[line][pixel].dc = 0;
            //else if(dx > max_dph/2) {
	    //  node_data[line][pixel].dc = (unsigned char)((1.0 - dx/max_dph) * (double)node_data[line][pixel].dc);
	    //}	
	  }
	}
      }
      else if(line < nrows - 1) {    // For middle rows ......
	if(pixel == 0) {
	  if(corr_data[line][pixel] > small && corr_data[line - 1][pixel] > small) {
	    dx = phase_data[line][pixel] - phase_data[line - 1][pixel];
	    dx = fabs(dx);
            if(dx > PI) dx = two_pi - dx;
	    if(dx >= max_dph) node_data[line][pixel].rc = 0;
            //else if(dx > max_dph/2) {
	    //  node_data[line][pixel].rc = (unsigned char)((1.0 - dx/max_dph) * (double)node_data[line][pixel].rc);
	    //}	
	  }
	}
	else if(pixel < ncols - 1) {
	  if(corr_data[line][pixel] > small && corr_data[line - 1][pixel] > small) {
	    dx = phase_data[line][pixel] - phase_data[line - 1][pixel];
	    dx = fabs(dx);
            if(dx > PI) dx = two_pi - dx;
	    if(dx >= max_dph) node_data[line][pixel].rc = 0;
            //else if(dx > max_dph/2) {
	    //  node_data[line][pixel].rc = (unsigned char)((1.0 - dx/max_dph) * (double)node_data[line][pixel].rc);
	    //}	
	  }
	  
	  if(corr_data[line][pixel] > small && corr_data[line][pixel - 1] > small) {
	    dx = phase_data[line][pixel] - phase_data[line][pixel - 1];
	    dx = fabs(dx);
            if(dx > PI) dx = two_pi - dx;
	    if(dx >= max_dph) node_data[line][pixel].dc = 0;
            //else if(dx > max_dph/2) {
	    //  node_data[line][pixel].dc = (unsigned char)((1.0 - dx/max_dph) * (double)node_data[line][pixel].dc);
	    //}	
	  }
	}
      }
	  
    }
  }

  if(0) {
    FILE *fp = fopen("dc.weight", "r");
    for(int i = 0; i < nrows; i ++) {
      for(int j = 0; j < ncols; j++) {
        fread(&node_data[i][j].dc, 1, 1, fp);
      }
    }
    fclose(fp);
    fp = fopen("rc.weight", "r");
    for(int i = 0; i < nrows; i ++) {
      for(int j = 0; j < ncols; j++) {
        fread(&node_data[i][j].rc, 1, 1, fp);
      }
    }
    fclose(fp);
  }

  if(0) {
    FILE *fp = fopen("test_dc.weight", "w");
    for(int i = 0; i < nrows; i ++) {
      for(int j = 0; j < ncols; j++) {
        fwrite(&node_data[i][j].dc, 1, 1, fp);
      }
    }
    fclose(fp);
    fp = fopen("test_rc.weight", "w");
    for(int i = 0; i < nrows; i ++) {
      for(int j = 0; j < ncols; j++) {
        fwrite(&node_data[i][j].rc, 1, 1, fp);
      }
    }
    fclose(fp);
    exit(0);
  }
  
// End of making nodes  !!!!!!!!!!!!!!!!!

// (2) seek minimum cost flow solution .......... 

  DataPatch<NodeFlow> *flows = solve(node_patch);
  NodeFlow **flow_data = flows->get_data_lines_ptr(); 

//  FILE *fp_flow = fopen("June20.int.flow", "r");
//  DataPatch<NodeFlow> *flows = new DataPatch<NodeFlow>(ncols, nrows);;
//  NodeFlow **flow_data = flows->get_data_lines_ptr();
//  for(int line = 0; line < nrows; line ++) {
//    fread(flow_data[line], 2, ncols, fp_flow);
//  }
//  fclose(fp_flow);

  if(corr_th > 0) {
    bool **isEdge = new bool * [nr_lines + 1];
    for(int line = 0; line < nr_lines + 1; line ++) {
      isEdge[line] = new bool[nr_pixels + 1];
      for(int pixel = 0; pixel < nr_pixels + 1; pixel ++) {
	isEdge[line][pixel] = false;
      }
    }
    uchar th = cost_scale * corr_th;
      cerr << "***** th: " << (int) th << endl;    
    for(int line = 0; line < nrows; line ++) {
      for(int pixel = 0; pixel < ncols; pixel ++) {
	if(node_data[line][pixel].rc < th && flow_data[line][pixel].toRight == 0) {
	  if(!isEdge[line][pixel]) flow_data[line][pixel].toRight = 1;
	}
	if(node_data[line][pixel].dc < th && flow_data[line][pixel].toDown == 0) {
	  if(!isEdge[line][pixel]) flow_data[line][pixel].toDown = 1;
	}
      }
    } 
    for(int line = 0; line < nr_lines + 1; line ++) {
      delete[] isEdge[line];
    }
    delete[] isEdge;
  }

// (3) start unwrap ..........
  
  int nr_seeds = 0;

  Seed *seeds = NULL;
  create_seeds(flows, min_pixels_per_region, nr_seeds, &seeds);      // create seeds only 
//  DataPatch<char> *visit_patch = unwrap_assp(flows, phase_data, nr_seeds, seeds); // unwrap only
  DataPatch<char> *visit_patch = unwrap_adjust_seeds(flows, phase_data, nr_seeds, seeds); // unwrap only
  delete visit_patch;
    
  generate_regions(flows, nr_seeds, seeds, region_map);

  
  delete[] seeds;
  delete flows;
  delete node_patch;
}

