// Copyright (c) 2017-, California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged.
// All rights reserved.
// 
// Author(s):
// 

/* ----------------------------------------------------------------------------
 * Author:  Xiaoqing Wu
 * April 2012 at JPL.
 * ************************************************************************** */
#include "CannyEdgeDetector.h"
#include "ChangeDetector.h"
#include "EdgeDetector.h"
#include "PhaseStatistics.h"
#include "ASSP.h"
#include "BMFS.h"
#include "Point.h"
#include "sort.h"
#include <omp.h> 

using namespace std;


void make_node_patch(char *int_file, char *corr_file, char *amp_file, char *layover_file,
		     int start_line, int nr_lines, int nr_pixels, double corr_th, double phase_th, double amp_th,
		     DataPatch<Node> **return_node_patch, DataPatch<float> **return_phase_patch, double minimum_corr,
			double good_corr, double const_phase_to_remove, int sq)
{
  // if corr_file does not exist, the magnitude of int_file is the correlation coefficients
  DataPatch<float> *phase_patch = new DataPatch<float>(nr_pixels, nr_lines);
  DataPatch<float> *corr_patch = new DataPatch<float>(nr_pixels, nr_lines);
  DataPatch<float> *amp_patch = NULL;
  double small = 0.000000001;

  float **phase_data = phase_patch->get_data_lines_ptr();
  float **corr_data = corr_patch->get_data_lines_ptr();
  float **amp_data = NULL;

  cerr << "make_node_patch :: \n";
  cerr << "int_file: " << int_file << endl;
  cerr << "corr_file: " << corr_file << endl;
  cerr << "amp_file: " << amp_file << endl;

  DataPatch<unsigned char> *Hweight_patch = NULL;
  DataPatch<unsigned char> *Vweight_patch = NULL;
  
  fcomplex ff(cos(const_phase_to_remove),-sin(const_phase_to_remove));

  if(fopen(corr_file, "r") == NULL) {  // no separate corr_file
    FILE *fp = fopen(int_file, "r");
    long long byte_offset = (long long)8 * (long long)nr_pixels * (long long)start_line;
    fseek(fp, byte_offset, SEEK_SET);
    fcomplex *int_data = new fcomplex[nr_pixels];
    for(int line = 0; line < nr_lines; line ++) {
      fread(int_data, 8, nr_pixels, fp);
      for(int pixel = 0; pixel < nr_pixels; pixel ++) {
	phase_data[line][pixel] = arg(int_data[pixel]*ff);
	corr_data[line][pixel] = abs(int_data[pixel]);
      }
    }
    fclose(fp);
    delete[] int_data;

    if(fopen(amp_file, "r") != NULL) {
      amp_patch = new DataPatch<float>(nr_pixels, nr_lines);
      amp_data = amp_patch->get_data_lines_ptr();
      fp = fopen(amp_file, "r");
      long long byte_offset = (long long)4 * (long long)nr_pixels * (long long)start_line;
      fseek(fp, byte_offset, SEEK_SET);
      for(int line = 0; line < nr_lines; line ++) {
	fread(amp_data[line], 4, nr_pixels, fp);
      }
      fclose(fp);
    }
  }
  else if(fopen(amp_file, "r") == NULL) {    // corr_file separate but amp_file not exist
    FILE *fp = fopen(corr_file, "r");
    long long byte_offset = (long long)4 * (long long)nr_pixels * (long long)start_line;
    fseek(fp, byte_offset, SEEK_SET);
    for(int line = 0; line < nr_lines; line ++) {
      fread(corr_data[line], 4, nr_pixels, fp);
    }
    fclose(fp);

    fp = fopen(int_file, "r");
    byte_offset = (long long)8 * (long long)nr_pixels * (long long)start_line;
    fseek(fp, byte_offset, SEEK_SET);
    amp_patch = new DataPatch<float>(nr_pixels, nr_lines);
    amp_data = amp_patch->get_data_lines_ptr();
    fcomplex *int_data = new fcomplex[nr_pixels];
    for(int line = 0; line < nr_lines; line ++) {
      fread(int_data, 8, nr_pixels, fp);
      for(int pixel = 0; pixel < nr_pixels; pixel ++) {
	phase_data[line][pixel] = arg(int_data[pixel]*ff);
	amp_data[line][pixel] = abs(int_data[pixel]);
	//if(corr_data[line][pixel] < small) {
	//  phase_data[line][pixel] = 0;
	//}
      }
    }
    fclose(fp);
    delete[] int_data;

  }
  else { // all 3 file exist ......
    FILE *fp = fopen(int_file, "r");
    long long byte_offset = (long long)8 * (long long)nr_pixels * (long long)start_line;
    fseek(fp, byte_offset, SEEK_SET);
    fcomplex *int_data = new fcomplex[nr_pixels];
    for(int line = 0; line < nr_lines; line ++) {
      fread(int_data, 8, nr_pixels, fp);
      for(int pixel = 0; pixel < nr_pixels; pixel ++) {
	phase_data[line][pixel] = arg(int_data[pixel]*ff);
      }
    }
    fclose(fp);
    delete[] int_data;

    
    fp = fopen(corr_file, "r");
    byte_offset = (long long)4 * (long long)nr_pixels * (long long)start_line;
    fseek(fp, byte_offset, SEEK_SET);
    for(int line = 0; line < nr_lines; line ++) {
      fread(corr_data[line], 4, nr_pixels, fp);
    }
    fclose(fp);
    
    amp_patch = new DataPatch<float>(nr_pixels, nr_lines);
    amp_data = amp_patch->get_data_lines_ptr();
    fp = fopen(amp_file, "r");
    byte_offset = (long long)4 * (long long)nr_pixels * (long long)start_line;
    fseek(fp, byte_offset, SEEK_SET);
    for(int line = 0; line < nr_lines; line ++) {
      fread(amp_data[line], 4, nr_pixels, fp);
    }
    fclose(fp);
  }

/*
  DataPatch<unsigned char> *edge_patch = NULL; 
  unsigned char **edge_data = NULL; 
  if(amp_data) {
    cerr << "update the costs with amplitude ...... \n";

    edge_patch = new DataPatch<unsigned char>(nr_pixels, nr_lines);
    edge_data = edge_patch->get_data_lines_ptr();

    int window_length = 7;
    double C_min = 0.5;  // 0.4
    double R_edge = 0.3; ! 0.2; // 0.30; // 0.4
    double R_line = 0.3; ! 0.1; //0.20; // 0.2

    cerr << "starting detect_edge() ...... \n";

    detect_edge(nr_lines, nr_pixels, amp_data, edge_data, window_length, C_min, R_edge, R_line);

    cerr << "ending detect_edge() ...... \n";
    
    for(int line = 0; line < nr_lines; line ++) {
      for(int pixel = 0; pixel < nr_pixels; pixel ++) {
        if(edge_data[line][pixel] > 0) corr_data[line][pixel] = 0;
      }
    }
  }
*/

  if(1) {
    if(1){
      for(int line = 0; line < nr_lines; line ++) {
        for(int pixel = 0; pixel < nr_pixels; pixel ++) {
          //double x = corr_data[line][pixel];
          //if(x > 0.6) corr_data[line][pixel] = 1.0;
          //corr_data[line][pixel] = x*x;
        }
      }
    }
  }
  else if(1) {
    DataPatch<unsigned char> *tmp_patch = new DataPatch<unsigned char>(nr_pixels, nr_lines);
    unsigned char **tmp_data = tmp_patch->get_data_lines_ptr();  
    double max_phase_std = 1.2;
    compute_corr(nr_lines, nr_pixels, phase_data, tmp_data, max_phase_std);

    for(int line = 0; line < nr_lines; line ++) {
      for(int pixel = 0; pixel < nr_pixels; pixel ++) {
        float x = (float)((double)tmp_data[line][pixel]/(double)cost_scale);
        if(corr_data[line][pixel] > 0) {
	  corr_data[line][pixel] = x;
	  //if(x < corr_data[line][pixel]) corr_data[line][pixel] = x;
          //if(corr_data[line][pixel] > 0.5) corr_data[line][pixel] = 0.5;
	}
      }
    }

    FILE *fp = fopen("tmp.cost", "w");
    for(int line = 0; line < nr_lines; line ++) {
      fwrite(corr_data[line], 4, nr_pixels, fp);
    }
    fclose(fp);

    delete tmp_patch;
  }
  else {
    Hweight_patch = compute_Hweight(nr_lines, nr_pixels, phase_data);
    Vweight_patch = compute_Hweight(nr_lines, nr_pixels, phase_data);
  }
  
/*
  // replace corr_data with the calculated phase noises ......
  if(0){
    DataPatch<unsigned char> *tmp_patch = new DataPatch<unsigned char>(nr_pixels, nr_lines);
    unsigned char **tmp_data = tmp_patch->get_data_lines_ptr();  
    double max_phase_std = 1.2;
    compute_corr(nr_lines, nr_pixels, phase_data, tmp_data, max_phase_std);


    for(int line = 0; line < nr_lines; line ++) {
      for(int pixel = 0; pixel < nr_pixels; pixel ++) {
        float x = (float)((double)tmp_data[line][pixel]/(double)cost_scale);
        double phase_deviation = 1.0 - x;
        double phase_std = phase_deviation + max_phase_std * edge_data[line][pixel]/255.0;
        //double phase_std = max_phase_std * edge_data[line][pixel]/255.0;
        if(phase_std > max_phase_std) phase_std = max_phase_std;
        corr_data[line][pixel] = 1.0 - phase_std / max_phase_std;
      }
    }

    delete tmp_patch;
  }
  else { 
    double max_phase_std = 7.0;
    for(int line = 0; line < nr_lines; line ++) {
      for(int pixel = 0; pixel < nr_pixels; pixel ++) {
        double x = corr_data[line][pixel];

//        x = sqrt((1.0 - x*x)/2.0)/x;
//        if(x == 0.0) x = 1000.0;
//        else x = 1.0/x;

//        x = 2.0*x - 1.0*x*x;
//	if(x > 1.0) x = 1.0;

        x = 1 - x;
        if(edge_data) {
	  if(edge_data[line][pixel] > 0) x = 1.0; // x += edge_data[line][pixel]/255.0;
 	}
        //double total = 1.0 - x + edge_data[line][pixel]/100.0;
        if(x > 1.0) x = 1.0;
        corr_data[line][pixel] = 1.0 - x;
      }
    }
  }
  if(edge_patch)  delete edge_patch;
*/
/*
  FILE *fp1 = fopen("tmp_corr.dat", "w");
  for(int line = 0; line < nr_lines; line ++) {
    fwrite(corr_data[line], 4, nr_pixels, fp1);
  }
  fclose(fp1);
  exit(0);
*/

 
  FILE *fp_lay = fopen(layover_file, "r");
  if(fp_lay) {  // yes layover_file
    unsigned char Zero = (unsigned char)0;
    long long byte_offset = (long long)sizeof(char) * (long long)nr_pixels * (long long)start_line;
    fseek(fp_lay, byte_offset, SEEK_SET);

    DataPatch<unsigned char> *layover_patch = new DataPatch<unsigned char>(nr_pixels, nr_lines);
    unsigned char **layover_data = layover_patch->get_data_lines_ptr();
    for(int line = 0; line < nr_lines; line ++) {
      fread(layover_data[line], sizeof(char), nr_pixels, fp_lay);
    }
    for(int line = 0; line < nr_lines; line ++) {
      for(int pixel = 0; pixel < nr_pixels; pixel ++) {
	if(layover_data[line][pixel] != Zero) {
	  if(pixel > 0) {
	    if(layover_data[line][pixel - 1] == Zero) {
	      corr_data[line][pixel] = 0.0;
	      continue;
	    }
	  }
	  if(pixel < nr_pixels - 1) {
	    if(layover_data[line][pixel + 1] == Zero) {
	      corr_data[line][pixel] = 0.0;
	      continue;
	    }
	  }
	  if(line > 0) {
	    if(layover_data[line - 1][pixel] == Zero) {
	      corr_data[line][pixel] = 0.0;
	      continue;
	    }
	  }
	  if(line < nr_lines - 1) {
	    if(layover_data[line + 1][pixel] == Zero) {
	      corr_data[line][pixel] = 0.0;
	      continue;
	    }
	  }
	}
      }
    }
    delete layover_patch;
/*
    unsigned char *byte_data = new unsigned char[nr_pixels];
    for(int line = 0; line < nr_lines; line ++) {
      //fread(byte_data, sizeof(char), nr_pixels, fp_lay);
      fread(layover_data, sizeof(char), nr_pixels, fp_lay);
      for(int pixel = 0; pixel < nr_pixels; pixel ++) {
	if(byte_data[pixel] != Zero) {
	  if(pixel > 0) {
	    if(
	  corr_data[line][pixel] = 0.0;
	}
      }
    }
    delete[] byte_data;
*/
    fclose(fp_lay);
  }

  if(0 && minimum_corr > 0.0) {
    for(int line = 0; line < nr_lines; line ++) {
      for(int pixel = 0; pixel < nr_pixels; pixel ++) {
	if(corr_data[line][pixel] < minimum_corr) {
	  corr_data[line][pixel] = 0.0;
	  // phase_data[line][pixel] = 0.0;
	}
      }
    }
  }
/*
  if(amp_data) {
    float low = 4.0;
    float high = 12.0;
    float mid = 8.0;
    int gw = 7;
    double gws = 1.0;

    for(int i = 0; i < nr_lines; i++) {
      for(int j = 0; j < nr_pixels; j++) {
        amp_data[i][j] = 10.0*log10(amp_data[i][j]+1.0e-20);  // for power inputs
      }
    }
    CannyEdgeDetector *edgeDetector = new CannyEdgeDetector(nr_lines, nr_pixels, -200.0, amp_data, low, high, gw, gws);

    float **edge = edgeDetector->get_edge();
 
    
    float corr_th_plus = corr_th + 1.01/(200.0 * corr_th); // 0.05; // corr_th + (sqrt(corr_th*corr_th + 0.01) - corr_th) + 0.01;

    cerr << "corr_th: " << corr_th << "  corr_th_plus: " << corr_th_plus << endl;

    for(int i = 0; i < nr_lines; i ++) {
      for(int j = 0; j < nr_pixels; j ++) {
	//if(edge[i][j] > low) corr_data[i][j] = 0;
	if(edge[i][j] > low && corr_data[i][j] > corr_th_plus) corr_data[i][j] = corr_th_plus;
	if(edge[i][j] >= mid) corr_data[i][j] = 0;
      }
    }
    delete edgeDetector;
  }
*/

  if(sq == 1) {
    for(int line = 0; line < nr_lines; line ++) {
      for(int pixel = 0; pixel < nr_pixels; pixel ++) {
	corr_data[line][pixel] *= corr_data[line][pixel];
      }
    }
  }


  

  cerr << "good_corr: " << good_corr << endl;
  cerr << "File read !!!! \n";


  //  

  int nrows = nr_lines + 1;
  int ncols = nr_pixels + 1;

  DataPatch<Node> *node_patch = new DataPatch<Node>(ncols, nrows);
  Node **node_data = node_patch->get_data_lines_ptr();

  for(int row = 0; row < nrows; row++) {
    for(int col = 0; col < ncols; col ++) {
      node_data[row][col].supply = 0;
      node_data[row][col].rc = 0;	  
      node_data[row][col].dc = 0;
//      node_data[row][col].edge_flag = edge_all;
    }
  }
  
  
  double pi = 3.14159265;
  double two_pi = 2.0 * pi;
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
/*
  double phase_difference_threshold = phase_th * pi;
  double dx = 0, dy = 0;

  
  double mean_phx = 0;
  double mean_phy = 0;
  double sigma_phx = 0;
  double sigma_phy = 0;
  int N_phx = 0;
  int N_phy = 0;

  for(int line = 0; line < nr_lines - 1; line ++) {
    for(int pixel = 0; pixel < nr_pixels - 1; pixel ++) {
      if(corr_data[line][pixel] > small) {
	if(corr_data[line][pixel + 1] > small) {
	  double x = phase_data[line][pixel + 1] - phase_data[line][pixel];
	  if(x > pi) x -= two_pi;
	  if(x < -pi) x += two_pi;
	  mean_phx += x;
	  sigma_phx += x*x;
	  N_phx ++;
	}
	if(corr_data[line + 1][pixel] > small) {
	  double x = phase_data[line + 1][pixel] - phase_data[line][pixel];
	  if(x > pi) x -= two_pi;
	  if(x < -pi) x += two_pi;
	  mean_phy += x;
	  sigma_phy += x*x;
	  N_phy ++;
	}
      }
    }
  }

  mean_phx /= (double)N_phx;
  mean_phy /= (double)N_phy;
  sigma_phx /= (double)N_phx;
  sigma_phy /= (double)N_phy;

  sigma_phx = sqrt(sigma_phx - mean_phx * mean_phx);
  sigma_phy = sqrt(sigma_phy - mean_phy * mean_phy);

  cerr << "mean_phx: " << mean_phx << "  sigma_phx: " << sigma_phx << endl;
  cerr << "mean_phy: " << mean_phy << "  sigma_phy: " << sigma_phy << endl;
  
*/
//  double corr_scale = sqrt(1.0 - corr_th*corr_th) / corr_th;
//  uchar cost_cap = 200;

/*
  for(int line = 0; line < nrows; line++) {
    for(int pixel = 0; pixel < ncols; pixel ++) {
      if(line == 0) {        // For the first row ......
	if(pixel > 0 && pixel < ncols - 1) {
	  node_data[line][pixel].dc = Vweight_patch->get_data_lines_ptr()[line][pixel - 1];
	}
      }
      else if(line < nrows - 1) {    // For middle rows ......
	node_data[line][pixel].rc = Hweight_patch->get_data_lines_ptr()[line - 1][pixel];
        if(pixel > 0 && pixel < ncols - 1) {
	  node_data[line][pixel].dc = Vweight_patch->get_data_lines_ptr()[line - 1][pixel - 1];
	}
      }
	  
    }
  }
*/  // Oct 21, 2015 commented by XWU

//  double tmp_corr_th = 0.5;
//  int cost_scale2 = cost_scale;// * cost_scale;

  int mask_th = good_corr * cost_scale;
  for(int line = 0; line < nrows; line++) {
    for(int pixel = 0; pixel < ncols; pixel ++) {
      if(line == 0) {        // For the first row ......
	if(pixel > 0 && pixel < ncols - 1) {
	  x = min(corr_data[line][pixel - 1], corr_data[line][pixel]);
	  node_data[line][pixel].dc = (uchar)(x * cost_scale);
	  //node_data[line][pixel].dc = 255;
	}
      }
      else if(line < nrows - 1) {    // For middle rows ......
	if(pixel == 0) {
	  y = min(corr_data[line - 1][pixel], corr_data[line][pixel]);
	  node_data[line][pixel].rc = (uchar)(y * cost_scale);
	  //node_data[line][pixel].rc = 255;
	}
//	else if(pixel == ncols - 2) {
//	  node_data[line][pixel].rc = 255;
//	  x = min(corr_data[line][pixel - 1], corr_data[line][pixel]);
//	  node_data[line][pixel].dc = (uchar)(x * cost_scale);
//	}
//	else if(pixel < ncols - 2) {
	else {
	  y = min(corr_data[line - 1][pixel], corr_data[line][pixel]);
	  node_data[line][pixel].rc = (uchar)(y * cost_scale);
	  x = min(corr_data[line][pixel - 1], corr_data[line][pixel]);
	  node_data[line][pixel].dc = (uchar)(x * cost_scale);

          //if(line == nrows - 2) { 
          //  node_data[line][pixel].dc = 255;
	  //}
	}
      }

      if(node_data[line][pixel].dc > mask_th) node_data[line][pixel].dc = 255;
      if(node_data[line][pixel].rc > mask_th) node_data[line][pixel].rc = 255;	  
    }
  }


  if(amp_data) {
    float low = 4.0;
    float high = 12.0;
    float mid = 8.0;
    int gw = 7;
    double gws = 1.0;

    for(int i = 0; i < nr_lines; i++) {
      for(int j = 0; j < nr_pixels; j++) {
        amp_data[i][j] = 10.0*log10(amp_data[i][j]+1.0e-20);  // for power inputs
      }
    }
    CannyEdgeDetector *edgeDetector = new CannyEdgeDetector(nr_lines, nr_pixels, -200.0, amp_data, low, high, gw, gws);

    float **edge = edgeDetector->get_edge();

    for(int i = 1; i < nr_lines - 1; i ++) {
      for(int j = 1; j < nr_pixels - 1; j ++) {
	if(edge[i][j] >= mid) {
	  //node_data[i][j].rc = 0;
	  //node_data[i][j].dc = 0;

	  if(edge[i + 1][j] >= mid) node_data[i][j].dc = 0;
	  if(edge[i][j + 1] >= mid) node_data[i][j].rc = 0;
          if(edge[i - 1][j] >= mid) node_data[i - 1][j].dc = 0;
	  if(edge[i][j - 1] >= mid) node_data[i][j - 1].rc = 0;
/*
	  if(edge[i - 1][j - 1] >= mid) {
	    node_data[i - 1][j - 1].dc = 0;
	    node_data[i][j - 1].rc = 0;
	  }
	  if(edge[i + 1][j - 1] >= mid) {
	    node_data[i + 1][j - 1].rc = 0;
	    node_data[i][j].dc = 0;
	  }
	  if(edge[i - 1][j + 1] >= mid) {
	    node_data[i - 1][j + 1].dc = 0;
	    node_data[i][j].rc = 0;
	  }
	  if(edge[i + 1][j + 1] >= mid) {
	    node_data[i + 1][j].rc = 0;
	    node_data[i][j].dc = 0;
	  }
*/
	  //if(edge[i + 1][j] >= low) node_data[i][j].rc = 0;
	  //if(edge[i][j + 1] >= low) node_data[i][j].dc = 0;
	}
      }
    }
  FILE *fp = fopen("edge.dat", "w");
    for(int ii = 0; ii < nr_lines; ii++) {
	fwrite(edge[ii], 4, nr_pixels, fp);
    }
  fclose(fp);
    delete edgeDetector;
  }

{
  FILE *fp = fopen("right_cost.dat", "w");
    for(int ii = 0; ii < nrows; ii++) {
      for(int jj = 0; jj < ncols; jj ++) {
        fwrite(&(node_data[ii][jj].rc), 1, 1, fp);
      }
    }
  fclose(fp);

 fp = fopen("down_cost.dat", "w");
    for(int ii = 0; ii < nrows; ii++) {
      for(int jj = 0; jj < ncols; jj ++) {
        fwrite(&(node_data[ii][jj].dc), 1, 1, fp);
      }
    }
  fclose(fp);

}

/*
  cerr << "correlation .........\n";
  for(int line = 0; line < nr_lines; line ++) {
    for(int pixel = 0; pixel < nr_pixels; pixel ++) {
      cerr << corr_data[line][pixel] << " ";
    }
    cerr << endl;
  }
  cerr << endl;


  cerr << "right cost .......... \n";
  for(int ii = 0; ii < nrows; ii++) {
    cerr << " " << ii % 10 << "  ";
    for(int jj = 0; jj < ncols; jj ++) {
      cerr << " " << (int)node_data[ii][jj].rc << " ";
    }
    cerr << endl;
  }
  cerr << endl;

  cerr << "down cost .......... \n";
  for(int ii = 0; ii < nrows; ii++) {
    cerr << " " << ii % 10 << "  ";
    for(int jj = 0; jj < ncols; jj ++) {
      cerr << " " << (int)node_data[ii][jj].dc << " ";
    }
    cerr << endl;
  }
  cerr << endl;
*/

// scale the rc (right cohearence) based on the pixel index, near or far range
/*
  for(int line = 0; line < nrows; line++) {
    for(int pixel = 0; pixel < ncols; pixel ++) {
      int rc = node_data[line][pixel].rc * 2.0 *pixel/(double)ncols;
      if(rc > 255) rc = 255;
      node_data[line][pixel].rc = (unsigned char)rc;
    }
  }
*/

  // update the costs with amplitude ......

/*
  if(amp_data) {
    cerr << "update the costs with amplitude ...... \n";
    
    int window_length = 5;
    double Ca_min = 0.5;
    double edge_ratio_max = 0.6;

    DataPatch<unsigned char> *Hedge_patch = new DataPatch<unsigned char>(nr_pixels, nr_lines);
    unsigned char **Hedge_data = Hedge_patch->get_data_lines_ptr();

    DataPatch<unsigned char> *Vedge_patch = new DataPatch<unsigned char>(nr_pixels, nr_lines);
    unsigned char **Vedge_data = Vedge_patch->get_data_lines_ptr();

    detect_edge(nr_lines, nr_pixels, amp_data, Hedge_data, Vedge_data, window_length, Ca_min, edge_ratio_max);

    double edge_th = 0;

    for(int line = 0; line < nrows; line++) {
      for(int pixel = 0; pixel < ncols; pixel ++) {
	if(line == 0) {        // For the first row ......
	  if(pixel > 0 && pixel < ncols - 1) {
            if(Vedge_data[line][pixel - 1] > edge_th) node_data[line][pixel].dc = i_min(10, node_data[line][pixel].dc);
	    //int r = node_data[line][pixel].dc - Vedge_data[line][pixel - 1] * 255/100.0; // *(1.0 - Vedge_data[line][pixel - 1]/255.0);
	    //if(r < 0) r = 0;
            //node_data[line][pixel].dc = (unsigned char)r;
	  }
	}
	else if(line < nrows - 1) {    // For middle rows ......
	  if(pixel == 0) {
	    if(Hedge_data[line - 1][pixel] > edge_th) node_data[line][pixel].rc = i_min(node_data[line][pixel].rc, 10);
	    //int r = node_data[line][pixel].rc - Hedge_data[line - 1][pixel] * 255/100.0; // *(1.0 - Hedge_data[line - 1][pixel]/255.0); 
	    //if(r < 0) r = 0;
	    //node_data[line][pixel].rc = (unsigned char)r;
	  }
	  else if(pixel < ncols - 1) {
	    if(Hedge_data[line - 1][pixel] > edge_th) node_data[line][pixel].rc = i_min(node_data[line][pixel].rc, 10);
	    //int r = node_data[line][pixel].rc - Hedge_data[line - 1][pixel] * 255/100.0; // *(1.0 - Hedge_data[line - 1][pixel]/255.0); 
	    //if(r < 0) r = 0;
	    //node_data[line][pixel].rc = (unsigned char)r;

            if(Hedge_data[line][pixel - 1] > edge_th) node_data[line][pixel].dc = i_min(node_data[line][pixel].dc, 10);
	   // r = node_data[line][pixel].dc - Vedge_data[line][pixel - 1] *255/100.0; // * (1.0 - Vedge_data[line][pixel - 1]/255.0); 

//if(abs(line - 1108) < 3 && abs(pixel - 936) < 3) cerr << "line: " << line << "  r: " << r << "  dc: " << (int)node_data[line][pixel].dc << "  Vedge: " << (int)Vedge_data[line][pixel - 1] << endl;
	    //if(r < 0) r = 0;
	    //node_data[line][pixel].dc = (unsigned char)r;	    
	  }
	}	
      }
    }
   

    delete Hedge_patch;
    delete Vedge_patch;
  }
*/

 
/*
int *cost = new int[ncols];
FILE *fp = fopen("tmp.Hedge", "w");
for(int line = 0; line < nrows; line ++) {
  for(int pixel = 0; pixel < ncols; pixel ++) {
    cost[pixel] = node_data[line][pixel].rc;
  }
  fwrite(cost, 4, ncols, fp);
}
fclose(fp);
fp = fopen("tmp.Vedge", "w");
for(int line = 0; line < nrows; line ++) {
  for(int pixel = 0; pixel < ncols; pixel ++) {
    cost[pixel] = node_data[line][pixel].dc;
  }
  fwrite(cost, 4, ncols, fp);
}
fclose(fp);
delete[] cost;
*/

  /*   old way 

  if(amp_data) {
    cerr << "update the costs with amplitude ...... \n";

    double dx, dy;
    
    DataPatch<float> *tmp_patch = new DataPatch<float>(nr_pixels, nr_lines); 
    float **tmp_data = tmp_patch->get_data_lines_ptr();
    for(int line = 0; line < nr_lines; line ++) {
      for(int pixel = 0; pixel < nr_pixels; pixel ++) {
	tmp_data[line][pixel] = amp_data[line][pixel];
      }
    }

    int win = 2;
    int len = 2*win + 1;

# pragma omp parallel for
    for(int line = win; line < nr_lines - win; line ++) {
      double *data = new double[len * len];
      int *indexes = new int[len * len];
      for(int pixel = win; pixel < nr_pixels - win; pixel ++) {
	int count = 0;
        for(int ii = line - win; ii <= line + win; ii++) {
	  for(int jj = pixel - win; jj <= pixel + win; jj ++) {
	    indexes[count] = count;
	    data[count] = tmp_data[ii][jj];
            count ++;
	  }
	}
        heapSort (count, data, indexes);
        amp_data[line][pixel] = data[indexes[count/2]];
      }    
      delete[] data;
      delete[] indexes;
    }
    delete tmp_patch;

    
    
    cerr << "amp_th: " << amp_th << endl;
    
    for(int line = 0; line < nrows; line++) {
      for(int pixel = 0; pixel < ncols; pixel ++) {
	//cerr << "line: " << line << "  pixel: " << pixel << endl;
	
	if(line == 0) {        // For the first row ......
	  if(pixel > 0 && pixel < ncols - 1) {
	    if(amp_data[line][pixel] > small && amp_data[line][pixel - 1] > small) {
	      dx = fabs(20.0*(log10(amp_data[line][pixel]) - log10(amp_data[line][pixel - 1])));
	      
	      //cerr << "line: " << line << "  pixel: " << pixel << "  dx: " << dx << endl;
	      
	      if(dx >= amp_th) node_data[line][pixel].dc = 0;
	      //else node_data[line][pixel].dc = (uchar)(node_data[line][pixel].dc * (amp_th - dx)/amp_th);
	    }
	  }
	}
	else if(line < nrows - 1) {    // For middle rows ......
	  if(pixel == 0) {
	    if(amp_data[line][pixel] > small && amp_data[line - 1][pixel] > small) {
	      dy = fabs(20.0*(log10(amp_data[line][pixel]) - log10(amp_data[line - 1][pixel])));
	      //cerr << "line: " << line << "  pixel: " << pixel << "  dy: " << dy << endl;
	      if(dy >= amp_th) node_data[line][pixel].rc = 0;
	      //else node_data[line][pixel].rc = (uchar)(node_data[line][pixel].rc * (amp_th - dy)/amp_th);
	    }
	  }
	  else if(pixel < ncols - 1) {
	    if(amp_data[line][pixel] > small && amp_data[line - 1][pixel] > small) {
	      dy = fabs(20.0*(log10(amp_data[line][pixel]) - log10(amp_data[line - 1][pixel])));
	      //cerr << "line: " << line << "  pixel: " << pixel << "  dy: " << dy << endl;
	      if(dy >= amp_th) node_data[line][pixel].rc = 0;
	      //else node_data[line][pixel].rc = (uchar)(node_data[line][pixel].rc * (amp_th - dy)/amp_th);
	    }
	    
	    if(amp_data[line][pixel] > small && amp_data[line][pixel - 1] > small) {
	      dx = fabs(20.0*(log10(amp_data[line][pixel]) - log10(amp_data[line][pixel - 1])));
	      //cerr << "line: " << line << "  pixel: " << pixel << "  dx: " << dx << endl;
	      if(dx >= amp_th) node_data[line][pixel].dc = 0;
	      //else node_data[line][pixel].dc = (uchar)(node_data[line][pixel].dc * (amp_th - dx)/amp_th);
	    }
	  }
	}
	
      }
    }
    
    cerr << "end of amp \n";
  }
*/



  // update the costs with phases ......
  
  double max_dph = phase_th; //1.0; // PI/2.0;
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
//            else {
//	      node_data[line][pixel].dc = (unsigned char)((1.0 - dx/max_dph) * (double)node_data[line][pixel].dc);
//	    }
	    //if(dx > phase_difference_threshold) dx = phase_difference_threshold;
	    //node_data[line][pixel].dc = (uchar)(x * (phase_difference_threshold - dx)/phase_difference_threshold * cost_scale);	
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
//            else {
//	      node_data[line][pixel].rc = (unsigned char)((1.0 - dx/max_dph) * (double)node_data[line][pixel].rc);
//	      //node_data[line][pixel].rc *= 1.0 - dx/max_dph;
//	    }
	    //if(dy > phase_difference_threshold) dy = phase_difference_threshold;
	    //node_data[line][pixel].rc = (uchar)(y * (phase_difference_threshold - dy)/phase_difference_threshold *cost_scale);
	  }
	}
	else if(pixel < ncols - 1) {
	  if(corr_data[line][pixel] > small && corr_data[line - 1][pixel] > small) {
	    dx = phase_data[line][pixel] - phase_data[line - 1][pixel];
	    dx = fabs(dx);
            if(dx > PI) dx = two_pi - dx;
	    if(dx >= max_dph) node_data[line][pixel].rc = 0;
//            else {
//	      node_data[line][pixel].rc = (unsigned char)((1.0 - dx/max_dph) * (double)node_data[line][pixel].rc);
//	    }
            //else node_data[line][pixel].rc *= 1.0 - dx/max_dph;
	  }
	  
	  if(corr_data[line][pixel] > small && corr_data[line][pixel - 1] > small) {
	    dx = phase_data[line][pixel] - phase_data[line][pixel - 1];
	    dx = fabs(dx);
            if(dx > PI) dx = two_pi - dx;
	    if(dx >= max_dph) node_data[line][pixel].dc = 0;
            //else node_data[line][pixel].dc *= 1.0 - dx/max_dph;
//            else {
//	      node_data[line][pixel].dc = (unsigned char)((1.0 - dx/max_dph) * (double)node_data[line][pixel].dc);
//	    }
	  }
	}
      }
	  
    }
  }

/*
  if(1) {
    FILE *fp = fopen("dc.weight", "w");
    for(int i = 0; i < nrows; i ++) {
      for(int j = 0; j < ncols; j++) {
        fwrite(&node_data[i][j].dc, 1, 1, fp);
      }
    }
    fclose(fp);
    fp = fopen("rc.weight", "w");
    for(int i = 0; i < nrows; i ++) {
      for(int j = 0; j < ncols; j++) {
        fwrite(&node_data[i][j].rc, 1, 1, fp);
      }
    }
    fclose(fp);
    exit(0);
  }
*/  


  delete corr_patch;
  if(amp_patch) delete amp_patch;

  if(Hweight_patch) delete Hweight_patch;
  if(Vweight_patch) delete Vweight_patch;

  *return_node_patch = node_patch;
  *return_phase_patch = phase_patch;

  cerr << "End of make_node_patch()  !!!!!! \n";

  // delete phase_patch;
  // return node_patch;
}

DataPatch<Node> *make_node_patch(DataPatch<fcomplex> *int_patch, double qthresh)
{
  int nr_lines = int_patch->get_nr_lines();
  int nr_pixels = int_patch->get_nr_pixels();
  // int nrows = nr_lines + 1;
  // int ncols = nr_pixels + 1;

  DataPatch<float> *phase_patch = new DataPatch<float>(nr_pixels, nr_lines);
  DataPatch<float> *amp_patch = new DataPatch<float>(nr_pixels, nr_lines);
  
  fcomplex **int_data = int_patch->get_data_lines_ptr();
  float **phase_data = phase_patch->get_data_lines_ptr();
  float **amp_data = amp_patch->get_data_lines_ptr();

  for(int line = 0; line < nr_lines; line ++) {
    for(int pixel = 0; pixel < nr_pixels; pixel ++) {
      phase_data[line][pixel] = arg(int_data[line][pixel]);
      amp_data[line][pixel] = abs(int_data[line][pixel]);
    }
  }

  DataPatch<Node> *node_patch = make_node_patch(nr_lines, nr_pixels, amp_data, phase_data, qthresh);

  delete amp_patch;
  delete phase_patch;

  return node_patch;
}


DataPatch<Node> *make_node_patch(int nr_lines, int nr_pixels, float **corr_data, float **phase_data, double qthresh)
{
  int nrows = nr_lines + 1;
  int ncols = nr_pixels + 1;

  DataPatch<Node> *node_patch = new DataPatch<Node>(ncols, nrows);
  Node **node_data = node_patch->get_data_lines_ptr();
  /*
  for(int line=0; line<nr_lines; line++) {
    for(int pixel=0; pixel<nr_pixels; pixel++) {
      if(corr_data[line][pixel] < qthresh) {
	corr_data[line][pixel] = 0.0;
      }
      corr_data[line][pixel] *= corr_data[line][pixel];
    }
  }
  */

  for(int row = 0; row < nrows; row++) {
    for(int col = 0; col < ncols; col ++) {
      node_data[row][col].supply = 0;
      node_data[row][col].rc = 0;	  
      node_data[row][col].dc = 0;
//      node_data[row][col].edge_flag = edge_all;
    }
  }
  
  double pi = 3.14159265;
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

    double phase_difference_threshold = 1.0*pi;
    double dx = 0, dy = 0;

  for(int line = 0; line < nrows; line++) {
    for(int pixel = 0; pixel < ncols; pixel ++) {
      if(line == 0) {        // For the first row ......
	if(pixel > 0 && pixel < ncols - 1) {
	  // x = (corr_data[line][pixel - 1] + corr_data[line][pixel] )/2.0;
	  x = min(corr_data[line][pixel - 1], corr_data[line][pixel]);
	  //x *= x;
	  //node_data[line][pixel].dc = (uchar)(x * cost_scale);	
	  dx = fabs(phase_data[line][pixel] - phase_data[line][pixel - 1]);
	  if(dx > pi) dx = 2*pi - dx;
	  if(dx > phase_difference_threshold) dx = phase_difference_threshold;
	  node_data[line][pixel].dc = (uchar)(x  * cost_scale);
	  //node_data[line][pixel].dc = (uchar)(x * (phase_difference_threshold - dx)/phase_difference_threshold * cost_scale);	
	}
      }
      else if(line < nrows - 1) {    // For middle rows ......
	if(pixel == 0) {
	  // y = ( corr_data[line - 1][pixel] + corr_data[line][pixel] )/2.0;
	  y = min(corr_data[line - 1][pixel], corr_data[line][pixel]);
	  //y *= y;
	  //node_data[line][pixel].rc = (uchar)(y *cost_scale);
	  dy = fabs(phase_data[line][pixel] - phase_data[line - 1][pixel]);
	  if(dy > pi) dy = 2*pi - dy;
	  if(dy > phase_difference_threshold) dy = phase_difference_threshold;
	  node_data[line][pixel].rc = (uchar)(y * cost_scale );
	  //node_data[line][pixel].rc = (uchar)(y * (phase_difference_threshold - dy)/phase_difference_threshold *cost_scale);
	}
	else if(pixel < ncols - 1) {
	  // y = (corr_data[line - 1][pixel] + corr_data[line][pixel] )/2.0;
	  y = min(corr_data[line - 1][pixel], corr_data[line][pixel]);
	  //y *= y;
	  //node_data[line][pixel].rc = (uchar)(y * cost_scale);
	  dy = fabs(phase_data[line][pixel] - phase_data[line - 1][pixel]);
	  if(dy > pi) dy = 2*pi - dy;
	  if(dy > phase_difference_threshold) dy = phase_difference_threshold;
	  //node_data[line][pixel].rc = (uchar)(y * (phase_difference_threshold - dy)/phase_difference_threshold * cost_scale);
	  node_data[line][pixel].rc = (uchar)(y * cost_scale);
	  // x = (corr_data[line][pixel - 1] + corr_data[line][pixel] )/2.0;

	  x = min(corr_data[line][pixel - 1], corr_data[line][pixel]);
	  //x *= x;
	  node_data[line][pixel].dc = (uchar)(x * cost_scale );
	  dx = fabs(phase_data[line][pixel] - phase_data[line][pixel - 1]);
	  if(dx > pi) dx = 2*pi - dx;
	  if(dx > phase_difference_threshold) dx = phase_difference_threshold;
	  node_data[line][pixel].dc = (uchar)(x * cost_scale);
	  //node_data[line][pixel].dc = (uchar)(x * (phase_difference_threshold - dx)/phase_difference_threshold * cost_scale );

	  // if(line % 100 == 0) cerr << line << " " << pixel << "  x: " << x << "  dc: " << (int)node_data[line][pixel].dc << "  cost_scale: " << cost_scale << endl;
	}
      }
	  
    }
  }


  /*
  for(int line = 0; line < nrows; line++) {
    for(int pixel = 0; pixel < ncols; pixel ++) {
      node_data[line][pixel].rc = 1;
      node_data[line][pixel].dc = 1;
      if(line == 0 || line == nrows - 1) {	
	node_data[line][pixel].rc = 0;
      }
      if(pixel == 0 || pixel == ncols - 1) {
	node_data[line][pixel].dc = 0;
      }
    }
  }
  */
  
  // Test Only **************
  /*
  for(int line = 0; line < nrows; line++) {
    for(int pixel = 0; pixel < ncols; pixel ++) {
      node_data[line][pixel].rc = 1;
      node_data[line][pixel].dc = 1;
      if(line == 0 || line == nrows - 1) {	
	node_data[line][pixel].rc = 0;
      }
      if(pixel == 0 || pixel == ncols - 1) {
	node_data[line][pixel].dc = 0;
      }
    }
  }
  */

  return node_patch;
}



DataPatch<char>* unwrap_assp(DataPatch<NodeFlow> *flows_patch, float **phase_data, int nr_seeds, Seed *seeds)
{
  int patch_start = flows_patch->get_extern_start_line();
  int nr_lines = flows_patch->get_nr_lines() - 1;
  int nr_pixels = flows_patch->get_nr_pixels() - 1;
  int line, pixel, line_plus, line_minus, pixel_plus, pixel_minus;
  //int nrows = nr_lines + 1;
  //int ncols = nr_pixels + 1;

  
  // do unwrapping with flood fill ......
  DataPatch<char> *visit_patch = new DataPatch<char>(nr_pixels, nr_lines);
  char **visit = visit_patch->get_data_lines_ptr();
  
  char not_unwrapped = 0;
  char unwrapped = 1;

  for(line = 0; line < nr_lines; line ++) {
    for(int pixel = 0; pixel < nr_pixels; pixel ++) {
      visit[line][pixel] = not_unwrapped;
    }
  }

  double two_pi = 2.0 * 3.14159265;
  double x, seed_phase;
  Point point, seed;

  NodeFlow **flows = flows_patch->get_data_lines_ptr();

  queue<Point> workq;
  
  for(int seed_id = 0; seed_id < nr_seeds; seed_id ++) {
    //  char unwrapped = seed_id + 1;

    int seed_x = seeds[seed_id].x;
    int seed_y = seeds[seed_id].y - patch_start;
    if(seed_y < 0 || seed_y >= nr_lines) continue;

//    if(visit[seed_y][seed_x] == unwrapped) continue;

    visit[seed_y][seed_x] = unwrapped;
    phase_data[seed_y][seed_x] += seeds[seed_id].nr_2pi * two_pi;

    workq.push(Point(seed_x, seed_y));

    // cerr << "Seed: " << Point(seed_x, seed_y) << "  nr_2pi: " << seeds[seed_id].nr_2pi << "  seed phase: " << phase_data[seed_y][seed_x] << endl;

    while( !workq.empty() ) {
      point = workq.front();
      workq.pop();
      line  = point.get_Y();
      pixel = point.get_X();
//      visit[line][pixel] = unwrapped;
      
      seed_phase = phase_data[line][pixel];
      
      line_plus = line + 1;
      line_minus = line - 1;
      pixel_plus = pixel + 1;
      pixel_minus = pixel - 1;
      
      if(line > 0) {              // facing up ......
	if(flows[line][pixel].toRight == 0 && visit[line_minus][pixel] == not_unwrapped) {
	  workq.push(Point(pixel, line_minus));
	  x = phase_data[line_minus][pixel] - seed_phase;
	  phase_data[line_minus][pixel] -= (int)(rint(x/two_pi)) * two_pi;
	  visit[line_minus][pixel] = unwrapped;
	}	
      }
      if(line < nr_lines - 1) {   // facing down ......    
	if(flows[line + 1][pixel].toRight == 0 && visit[line_plus][pixel] == not_unwrapped) {
	  workq.push(Point(pixel, line_plus));
	  x = phase_data[line_plus][pixel] - seed_phase;
	  phase_data[line_plus][pixel] -= (int)(rint(x/two_pi)) * two_pi;
	  visit[line_plus][pixel] = unwrapped;
	}	
      }
      if(pixel > 0) {             // facing left ......
	if(flows[line][pixel].toDown == 0 && visit[line][pixel_minus] == not_unwrapped) {
	  workq.push(Point(pixel_minus, line));
	  x = phase_data[line][pixel_minus] - seed_phase;
	  phase_data[line][pixel_minus] -= (int)(rint(x/two_pi)) * two_pi;
	  visit[line][pixel_minus] = unwrapped;
	}	
      }
      if(pixel < nr_pixels - 1) {// facing right ......
	if(flows[line][pixel_plus].toDown == 0 && visit[line][pixel_plus] == not_unwrapped) {
	  workq.push(Point(pixel_plus, line));
	  x = phase_data[line][pixel_plus] - seed_phase;
	  phase_data[line][pixel_plus] -= (int)(rint(x/two_pi)) * two_pi;
	  visit[line][pixel_plus] = unwrapped;
	}	
      }
    }
  }

  for(line = 0; line < nr_lines; line ++) {
    for(int pixel = 0; pixel < nr_pixels; pixel ++) {
      if(visit[line][pixel] == not_unwrapped) {
	phase_data[line][pixel] = no_data_value;
      }
    }
  }
  
  return visit_patch;
}

void create_seeds(DataPatch<NodeFlow> *flows_patch, int minimum_nr_pixels, int& nr_seeds, Seed **seeds)
{
  int nr_lines = flows_patch->get_nr_lines() - 1;
  int nr_pixels = flows_patch->get_nr_pixels() - 1;
  int line, pixel, line_plus, line_minus, pixel_plus, pixel_minus;
  int nrows = nr_lines + 1;
  int ncols = nr_pixels + 1;

  // cerr << "nr_lines: " << nr_lines << "  nr_pixels: " << nr_pixels << endl;
  // cerr << "nrows: " << nrows << "  ncols: " << ncols << endl;

  int max_seeds = 1000000;


  Seed *tmp_seeds = new Seed[max_seeds];
  double *pixels_of_region = new double[max_seeds];

  
  // do unwrapping with flood fill ......
  DataPatch<char> *visit_patch = new DataPatch<char>(ncols - 1, nrows - 1);
  char **visit = visit_patch->get_data_lines_ptr();
  
  char not_unwrapped = 0;
  char unwrapped = 1;

  for(line = 0; line < nrows - 1; line ++) {
    for(int pixel = 0; pixel < ncols - 1; pixel ++) {
      visit[line][pixel] = not_unwrapped;
    }
  }

  Point point, seed;

  NodeFlow **flows = flows_patch->get_data_lines_ptr();

  queue<Point> workq;

  int region_id = 0;

  //  cerr << "region_id: " << region_id << endl;
  
  for(int ii = 0; ii < nr_lines; ii++) {
    // cerr << "ii: " << ii << endl;
    //for(int jj = 0; jj < nr_pixels; jj++) {
    // reversing the order so that farther out pixesl are used for the seed, reducing the sensitivity
    // to bad dem
    for(int jj = nr_pixels-1; jj >=0; jj--) {
          
      // cerr << "ii: " << ii << "  jj: " << jj << "  visit: " << visit[jj][ii] << "  region_id: " << region_id << "  visit: " << (int)visit[ii][jj]<< endl;

      if(visit[ii][jj] == unwrapped) continue;



      int count = 0;
      workq.push(Point(jj, ii));

      visit[ii][jj] = unwrapped;

      tmp_seeds[region_id].x = jj;
      tmp_seeds[region_id].y = ii;
      tmp_seeds[region_id].nr_2pi = 0;
      
      while( !workq.empty() ) {
	point = workq.front();
	workq.pop();
	line  = point.get_Y();
	pixel = point.get_X();

	count ++;

        if(count == minimum_nr_pixels / 2) {	  
          tmp_seeds[region_id].x = pixel;
          tmp_seeds[region_id].y = line;
          tmp_seeds[region_id].nr_2pi = 0;
 	}

	
	line_plus = line + 1;
	line_minus = line - 1;
	pixel_plus = pixel + 1;
	pixel_minus = pixel - 1;
	
	if(line > 0) {              // facing up ......
	  if(flows[line][pixel].toRight == 0 && visit[line_minus][pixel] == not_unwrapped) {
	    workq.push(Point(pixel, line_minus));
	    visit[line_minus][pixel] = unwrapped;
	  }	
	}
	if(line < nr_lines - 1) {   // facing down ......    
	  if(flows[line_plus][pixel].toRight == 0 && visit[line_plus][pixel] == not_unwrapped) {
	    workq.push(Point(pixel, line_plus));
	    visit[line_plus][pixel] = unwrapped;
	  }	
	}
	if(pixel > 0) {             // facing left ......
	  if(flows[line][pixel].toDown == 0 && visit[line][pixel_minus] == not_unwrapped) {
	    workq.push(Point(pixel_minus, line));
	    visit[line][pixel_minus] = unwrapped;
	  }	
	}
	if(pixel < nr_pixels - 1) {// facing right ......
	  if(flows[line][pixel_plus].toDown == 0 && visit[line][pixel_plus] == not_unwrapped) {
	    workq.push(Point(pixel_plus, line));
	    visit[line][pixel_plus] = unwrapped;
	  }	
	}
      }
      // if(count > 1) cerr << "ii: " << ii << "  jj: " << jj << "  region_id: " << region_id << "  count: " << count << "  mini: " << minimum_nr_pixels<<  endl;
      if(count >= minimum_nr_pixels) {
	
	pixels_of_region[region_id] = count;
	region_id ++;
	
	if(region_id >= max_seeds) break;
      }
    }
    if(region_id >= max_seeds) break;

  }

  int nr_regions = region_id;

  // cerr << "nr_regions: " << nr_regions << endl;


  int *indexes = new int[nr_regions];  
  heapSort(nr_regions, pixels_of_region, indexes, 1);

  nr_seeds = nr_regions;
  /*
  for(int i = 0; i < 10; i++) {
    int index = indexes[i];
    cerr << "i: " << i << "  index: " << index << "  pixels_of_region: " << pixels_of_region[ index ] << "  seed: " << tmp_seeds[index].x << " : " << tmp_seeds[index].y << endl;
  }
  */

  Seed *return_seeds = new Seed[nr_seeds];

  for(int i = 0; i < nr_regions; i++) {
    int index = indexes[i];
    return_seeds[i].x = tmp_seeds[index].x;
    return_seeds[i].y = tmp_seeds[index].y;
    return_seeds[i].nr_2pi = tmp_seeds[index].nr_2pi;

    //  cerr << "i: " << i << "  index: " << index << "  seed: " << return_seeds[i].x << " " << return_seeds[i].y << " " << return_seeds[i].nr_2pi << "  pixels_of_region: " << pixels_of_region[index] << endl;
  }
  

  delete[] indexes;
  delete[] pixels_of_region;
  delete[] tmp_seeds;

  delete visit_patch;

  *seeds = return_seeds;
}


DataPatch<char>* unwrap_adjust_seeds(DataPatch<NodeFlow> *flows_patch, float **phase_data, int nr_seeds, Seed *seeds)
{
  int patch_start = flows_patch->get_extern_start_line();
  int nr_lines = flows_patch->get_nr_lines() - 1;
  int nr_pixels = flows_patch->get_nr_pixels() - 1;
  int line, pixel, line_plus, line_minus, pixel_plus, pixel_minus;
  //int nrows = nr_lines + 1;
  //int ncols = nr_pixels + 1;

  
  // do unwrapping with flood fill ......
  DataPatch<char> *visit_patch = new DataPatch<char>(nr_pixels, nr_lines);
  char **visit = visit_patch->get_data_lines_ptr();
  
  char not_unwrapped = 0;
  char unwrapped = 1;

  for(line = 0; line < nr_lines; line ++) {
    for(int pixel = 0; pixel < nr_pixels; pixel ++) {
      visit[line][pixel] = not_unwrapped;
    }
  }

  double two_pi = 2.0 * 3.14159265;
  double x, seed_phase;
  Point point, seed;

  NodeFlow **flows = flows_patch->get_data_lines_ptr();

  queue<Point> workq, backupq;
  
  int nr_amb = 21;
  int *histogram = new int[nr_amb];
  double *lower_bound = new double[nr_amb];
  double *upper_bound = new double[nr_amb];
  for(int i = 0; i < nr_amb; i++) {
    lower_bound[i] = -PI + two_pi * (i - nr_amb/2);
    upper_bound[i] = PI + two_pi * (i - nr_amb/2);
  }

  for(int seed_id = 0; seed_id < nr_seeds; seed_id ++) {
    int seed_x = seeds[seed_id].x;
    int seed_y = seeds[seed_id].y - patch_start;
    if(seed_y < 0 || seed_y >= nr_lines) continue;

    if(visit[seed_y][seed_x] == unwrapped) continue;

    phase_data[seed_y][seed_x] += seeds[seed_id].nr_2pi * two_pi;

    workq.push(Point(seed_x, seed_y));
    backupq.push(Point(seed_x, seed_y));

    for(int i = 0; i < nr_amb; i++) histogram[i] = 0;

    while( !workq.empty() ) {
      point = workq.front();
      workq.pop();
      line  = point.get_Y();
      pixel = point.get_X();
      visit[line][pixel] = unwrapped;
      
      seed_phase = phase_data[line][pixel];

      for(int aid = 0; aid < nr_amb; aid++) {
        if(seed_phase >= lower_bound[aid] && seed_phase < upper_bound[aid]) {
	  histogram[aid] ++;
	}
      }
      
      line_plus = line + 1;
      line_minus = line - 1;
      pixel_plus = pixel + 1;
      pixel_minus = pixel - 1;
      
      if(line > 0) {              // facing up ......
	if(flows[line][pixel].toRight == 0 && visit[line_minus][pixel] == not_unwrapped) {
	  workq.push(Point(pixel, line_minus));
	  backupq.push(Point(pixel, line_minus));
	  x = phase_data[line_minus][pixel] - seed_phase;
	  phase_data[line_minus][pixel] -= (int)(rint(x/two_pi)) * two_pi;
	  visit[line_minus][pixel] = unwrapped;
	}	
      }
      if(line < nr_lines - 1) {   // facing down ......    
	if(flows[line + 1][pixel].toRight == 0 && visit[line_plus][pixel] == not_unwrapped) {
	  workq.push(Point(pixel, line_plus));
	  backupq.push(Point(pixel, line_plus));
	  x = phase_data[line_plus][pixel] - seed_phase;
	  phase_data[line_plus][pixel] -= (int)(rint(x/two_pi)) * two_pi;
	  visit[line_plus][pixel] = unwrapped;
	}	
      }
      if(pixel > 0) {             // facing left ......
	if(flows[line][pixel].toDown == 0 && visit[line][pixel_minus] == not_unwrapped) {
	  workq.push(Point(pixel_minus, line));
	  backupq.push(Point(pixel_minus, line));
	  x = phase_data[line][pixel_minus] - seed_phase;
	  phase_data[line][pixel_minus] -= (int)(rint(x/two_pi)) * two_pi;
	  visit[line][pixel_minus] = unwrapped;
	}	
      }
      if(pixel < nr_pixels - 1) {// facing right ......
	if(flows[line][pixel_plus].toDown == 0 && visit[line][pixel_plus] == not_unwrapped) {
	  workq.push(Point(pixel_plus, line));
	  backupq.push(Point(pixel_plus, line));
	  x = phase_data[line][pixel_plus] - seed_phase;
	  phase_data[line][pixel_plus] -= (int)(rint(x/two_pi)) * two_pi;
	  visit[line][pixel_plus] = unwrapped;
	}	
      }
    }
    int N = 0;
    int histo_max = 0;
    for(int i = 0; i < nr_amb; i++) {
      if(histogram[i] > histo_max) {
	histo_max = histogram[i];
	N = i;
      }
    }

    N -= nr_amb/2;
//    cerr << "seed_id: " << seed_id << "  N: " << N << endl;

    if(N != 0) {
      seeds[seed_id].nr_2pi -= N;
      double phase_adjust = two_pi * N;
      while( !backupq.empty() ) {
        point = backupq.front();
        backupq.pop();
        line  = point.get_Y();
        pixel = point.get_X();
	phase_data[line][pixel] -= phase_adjust;
      }
    }
    else {
      while( !backupq.empty() ) {
        backupq.pop();
      }
    }
  }

  for(line = 0; line < nr_lines; line ++) {
    for(int pixel = 0; pixel < nr_pixels; pixel ++) {
      if(visit[line][pixel] == not_unwrapped) {
	phase_data[line][pixel] = no_data_value;
      }
    }
  }
  
  delete[] histogram;
  delete[] lower_bound;
  delete[] upper_bound;
  //delete visit_patch;

  return visit_patch;
}

DataPatch<int> * generate_regions(DataPatch<NodeFlow> *flows_patch, int nr_seeds, Seed *seeds)
{
  int patch_start = flows_patch->get_extern_start_line();
  int nr_lines = flows_patch->get_nr_lines() - 1;
  int nr_pixels = flows_patch->get_nr_pixels() - 1;
  int line, pixel, line_plus, line_minus, pixel_plus, pixel_minus;
  //int nrows = nr_lines + 1;
  //int ncols = nr_pixels + 1;

  
  // do unwrapping with flood fill ......
  DataPatch<int> *visit_patch = new DataPatch<int>(nr_pixels, nr_lines);
  int **visit = visit_patch->get_data_lines_ptr();
  
  int not_unwrapped = -1;

  for(line = 0; line < nr_lines; line ++) {
    for(int pixel = 0; pixel < nr_pixels; pixel ++) {
      visit[line][pixel] = not_unwrapped;
    }
  }

  Point point, seed;

  NodeFlow **flows = flows_patch->get_data_lines_ptr();

  queue<Point> workq;

  for(int seed_id = 0; seed_id < nr_seeds; seed_id ++) {

    int seed_x = seeds[seed_id].x;
    int seed_y = seeds[seed_id].y - patch_start;
    if(seed_y < 0 || seed_y >= nr_lines) continue;

    if(visit[seed_y][seed_x] != not_unwrapped) continue;

    workq.push(Point(seed_x, seed_y));

    while( !workq.empty() ) {
      point = workq.front();
      workq.pop();
      line  = point.get_Y();
      pixel = point.get_X();
      visit[line][pixel] = seed_id;
      
      
      line_plus = line + 1;
      line_minus = line - 1;
      pixel_plus = pixel + 1;
      pixel_minus = pixel - 1;
      
      if(line > 0) {              // facing up ......
	if(flows[line][pixel].toRight == 0 && visit[line_minus][pixel] == not_unwrapped) {
	  workq.push(Point(pixel, line_minus));
	  visit[line_minus][pixel] = seed_id;
	}	
      }
      if(line < nr_lines - 1) {   // facing down ......    
	if(flows[line + 1][pixel].toRight == 0 && visit[line_plus][pixel] == not_unwrapped) {
	  workq.push(Point(pixel, line_plus));
	  visit[line_plus][pixel] = seed_id;
	}	
      }
      if(pixel > 0) {             // facing left ......
	if(flows[line][pixel].toDown == 0 && visit[line][pixel_minus] == not_unwrapped) {
	  workq.push(Point(pixel_minus, line));
	  visit[line][pixel_minus] = seed_id;
	}	
      }
      if(pixel < nr_pixels - 1) {// facing right ......
	if(flows[line][pixel_plus].toDown == 0 && visit[line][pixel_plus] == not_unwrapped) {
	  workq.push(Point(pixel_plus, line));
	  visit[line][pixel_plus] = seed_id;
	}	
      }
    }
  }

  return visit_patch;
}



void generate_regions(DataPatch<NodeFlow> *flows_patch, int nr_seeds, Seed *seeds, int **regions)
{
  int patch_start = flows_patch->get_extern_start_line();
  int nr_lines = flows_patch->get_nr_lines() - 1;
  int nr_pixels = flows_patch->get_nr_pixels() - 1;
  int line, pixel, line_plus, line_minus, pixel_plus, pixel_minus;
  //int nrows = nr_lines + 1;
  //int ncols = nr_pixels + 1;

  
  // do unwrapping with flood fill ......
  
  int not_unwrapped = -1;

  for(line = 0; line < nr_lines; line ++) {
    for(int pixel = 0; pixel < nr_pixels; pixel ++) {
      regions[line][pixel] = not_unwrapped;
    }
  }

  Point point, seed;

  NodeFlow **flows = flows_patch->get_data_lines_ptr();

  queue<Point> workq;

  for(int seed_id = 0; seed_id < nr_seeds; seed_id ++) {

    int seed_x = seeds[seed_id].x;
    int seed_y = seeds[seed_id].y - patch_start;
    if(seed_y < 0 || seed_y >= nr_lines) continue;

    if(regions[seed_y][seed_x] != not_unwrapped) continue;

    workq.push(Point(seed_x, seed_y));

    while( !workq.empty() ) {
      point = workq.front();
      workq.pop();
      line  = point.get_Y();
      pixel = point.get_X();
      regions[line][pixel] = seed_id;
      
      
      line_plus = line + 1;
      line_minus = line - 1;
      pixel_plus = pixel + 1;
      pixel_minus = pixel - 1;
      
      if(line > 0) {              // facing up ......
	if(flows[line][pixel].toRight == 0 && regions[line_minus][pixel] == not_unwrapped) {
	  workq.push(Point(pixel, line_minus));
	  regions[line_minus][pixel] = seed_id;
	}	
      }
      if(line < nr_lines - 1) {   // facing down ......    
	if(flows[line + 1][pixel].toRight == 0 && regions[line_plus][pixel] == not_unwrapped) {
	  workq.push(Point(pixel, line_plus));
	  regions[line_plus][pixel] = seed_id;
	}	
      }
      if(pixel > 0) {             // facing left ......
	if(flows[line][pixel].toDown == 0 && regions[line][pixel_minus] == not_unwrapped) {
	  workq.push(Point(pixel_minus, line));
	  regions[line][pixel_minus] = seed_id;
	}	
      }
      if(pixel < nr_pixels - 1) {// facing right ......
	if(flows[line][pixel_plus].toDown == 0 && regions[line][pixel_plus] == not_unwrapped) {
	  workq.push(Point(pixel_plus, line));
	  regions[line][pixel_plus] = seed_id;
	}	
      }
    }
  }
}


DataPatch<NodeFlow> *solve(DataPatch<Node> *node_patch)
{
  int nrows = node_patch->get_nr_lines();
  int ncols = node_patch->get_nr_pixels();
  Node **nodes = node_patch->get_data_lines_ptr();
  
  Flow flow;
  
  DataPatch<NodeFlow> *node_flow_patch = new DataPatch<NodeFlow>(ncols, nrows);
  DataPatch<NodeFlow> *new_flow_patch = new DataPatch<NodeFlow>(ncols, nrows);
  NodeFlow **node_flow = node_flow_patch->get_data_lines_ptr();
  NodeFlow **new_flow  = new_flow_patch->get_data_lines_ptr();
  
  for(int line = 0; line < nrows; line++) {
    for(int pixel = 0; pixel < ncols; pixel ++) {
      node_flow[line][pixel].toRight = 0;
      node_flow[line][pixel].toDown = 0;
      new_flow[line][pixel].toRight = 0;
      new_flow[line][pixel].toDown = 0;
    }
  }
  

  //count the positive and negative residues 
  int pcount = 0;
  int ncount = 0;
  for(int line = 0; line < nrows; line++) {
    for(int pixel = 0; pixel < ncols; pixel ++) {
//      if(nodes[line][pixel].edge_flag == 0) continue;
      if(nodes[line][pixel].supply > 0)      pcount ++;
      else if(nodes[line][pixel].supply < 0) ncount ++;
    }
  }

  cerr << "pcount: " << pcount << "  ncount: " << ncount << endl; 

  int supplys, demands;
  char supply = 1;
  char demand = -1;
  supplys = pcount;
  demands = ncount;


  //int origin_supply = - (pcount - ncount);
  if(abs(pcount - ncount) == 1) nodes[0][0].supply = - (pcount - ncount);
  else if(abs(pcount - ncount) > 1) {
    char tmp_supply = 1;
    if(pcount > ncount) tmp_supply = -1;
    int count = abs(pcount - ncount);
    int index = 0;
    for(int line = 0; line < nrows; line ++) {
      int step = 1;
      if(line > 0 && line < nrows - 1) step = ncols - 1;
      for(int pixel = 0; pixel < ncols; pixel += step) {
	nodes[line][pixel].supply = tmp_supply;
	index ++;
	if(index >= count) break;
      }
      if(index >= count) break;
    }
	
  }


  if(pcount < ncount) {
    supplys = ncount;
    demands = ncount;
    supply = 1;
    demand = -1;
  }
  else {
    supplys = pcount;
    demands = pcount;
    supply = -1;
    demand = 1;
  }

  if(pcount < ncount) pcount = ncount;
  else if(pcount > ncount) ncount = pcount;

  cerr << "nodes[0][0].supply: " << (int)nodes[0][0].supply << endl;

  int line, pixel;

  cerr << "supply: " << (int)supply << "  supplys : " << supplys << endl;
  cerr << "demand: " << (int)demand << "  demands : " << demands << endl;

/*
  cerr << "Residues ............\n";
  cerr << "     ";  
  for(int ii = 0; ii < ncols; ii ++) {
    cerr << ii % 10 << "  ";
  }
  cerr << endl;
  
  for(int ii = 0; ii < nrows; ii++) {
    cerr << " " << ii % 10 << "  ";
    for(int jj = 0; jj < ncols; jj ++) {
      if(nodes[ii][jj].supply == supply) cerr << " + ";
      else if(nodes[ii][jj].supply == demand) cerr << " - ";
      else cerr << "   ";
    }
    cerr << endl;
  }
  cerr << endl;

  cerr << "right cost .......... \n";
  for(int ii = 0; ii < nrows; ii++) {
    cerr << " " << ii % 10 << "  ";
    for(int jj = 0; jj < ncols; jj ++) {
      cerr << " " << (int)nodes[ii][jj].rc << " ";
    }
    cerr << endl;
  }
  cerr << endl;

  cerr << "down cost .......... \n";
  for(int ii = 0; ii < nrows; ii++) {
    cerr << " " << ii % 10 << "  ";
    for(int jj = 0; jj < ncols; jj ++) {
      cerr << " " << (int)nodes[ii][jj].dc << " ";
    }
    cerr << endl;
  }
  cerr << endl;
*/

  // initialize the Supply terminal and the Demand terminal 

  
  Point *S = new Point[supplys];
  Point *T = new Point[supplys];
  pcount = 0;
  ncount = 0;
  
  for(int line = 0; line < nrows; line++) {
    for(int pixel = 0; pixel < ncols; pixel ++) {
//      if(nodes[line][pixel].edge_flag == 0) continue;
      if(nodes[line][pixel].supply == 0) continue;
      if(nodes[line][pixel].supply == supply) {
	S[pcount].x = pixel;
	S[pcount].y = line;
	pcount ++;
      }
      else {
	T[ncount].x = pixel;
	T[ncount].y = line;
	ncount ++;
      }
    }
  }
  

  demands = ncount;

  // Start the algorithm .......................................

  // (1) initialization of the node potentials and the initial flows ......

  DataPatch<uint> *pot_patch = new DataPatch<uint>(ncols, nrows);
  uint **pot = pot_patch->get_data_lines_ptr();
  for(int line = 0; line < nrows; line++) {
    for(int pixel = 0; pixel < ncols; pixel ++) {
      pot[line][pixel] = 0;
    }
  }

//  initialize_flows_PI(nrows, ncols, nodes, supplys, S, demands, T, node_flow, pot);
  
/*
  int new_supplys = 0;
  for(int i = 0; i < supplys; i++) {
    line = S[i].y;
    pixel = S[i].x;
    if(nodes[line][pixel].supply != 0) new_supplys ++;
  }
  
  int new_demands = 0;
  for(int i = 0; i < demands; i++) {
    line = T[i].y;
    pixel = T[i].x;
    if(nodes[line][pixel].supply != 0) new_demands ++;
  }

  Point *new_S = new Point[new_supplys];
  int index = 0;
  for(int i = 0; i < supplys; i++) {
    line = S[i].y;
    pixel = S[i].x;
    if(nodes[line][pixel].supply != 0) {
      new_S[index].y = line;
      new_S[index].x = pixel;

      //cerr << "i: " << i << "  index: " << index << "  supply: " << new_S[index] << endl;

      index ++;
    }
  }
    
  delete[] S;
  
  Point *new_T = new Point[new_demands];
  index = 0;
  for(int i = 0; i < demands; i++) {
    line = T[i].y;
    pixel = T[i].x;
    if(nodes[line][pixel].supply != 0) {
      new_T[index].y = line;
      new_T[index].x = pixel;
      //cerr << "i: " << i << "  index: " << index << "  demand: " << new_T[index] << endl;

      index ++;
    }
  }
    
  delete[] T;
*/
  

  // (2) initialization of the visit matrix and the distance matrix ......

  uint infinity = 1000000000;

  uchar unlabeled = (uchar)0;
  uchar labeled   = (uchar)1;
  uchar scanned   = (uchar)2;
  DataPatch<uchar> *visit_patch = new DataPatch<uchar>(ncols, nrows);
  uchar **visit = visit_patch->get_data_lines_ptr();


  DataPatch<uint> *dist_patch = new DataPatch<uint>(ncols, nrows);
  uint **dists = dist_patch->get_data_lines_ptr();
  

  DataPatch<uchar> *branch_patch = new DataPatch<uchar>(ncols, nrows);
  uchar **branches = branch_patch->get_data_lines_ptr();
  for(int ii = 0; ii < nrows; ii++) {
    for(int jj = 0; jj < ncols; jj ++) {
      branches[ii][jj] = noflow;
    }
  }

  // (3) find the shortest pathes from S to T

  // First scan the nodes in S and initialize them ......

  int nr_queues = cost_scale * min(ncols, nrows) * 2;

  queue<Point> *dist_queues = new queue<Point>[nr_queues];
  
  Point point;

  uint d, curr_dist, reduced_cost;

  
  // start loop from here ......

  int supplys_left = supplys;


//  int *indexes = new int[new_demands];
//  double *dd = new double[new_demands];
  int *indexes = new int[demands];
  double *dd = new double[demands];

  int iter = 0;
  while (supplys_left > 0) {

//    cerr << "\n iter: " << iter << endl;
    // (1) initializing ......
    
    for(int ii = 0; ii < nrows; ii++) {
      for(int jj = 0; jj < ncols; jj ++) {
	visit[ii][jj] = unlabeled;
	dists[ii][jj] = infinity;
      }
    }
    

//    for(int s = 0; s < new_supplys; s++) {
//      line = new_S[s].y;
//      pixel = new_S[s].x;
    for(int s = 0; s < supplys; s++) {
      line = S[s].y;
      pixel = S[s].x;
      if(nodes[line][pixel].supply == 0) continue;   // if Residue discharged
      dists[ line ][ pixel ] = 0;  // Otherwise set all left-over supplys to zero distance
      visit[line][pixel] = labeled;
      point.x = pixel;
      point.y = line;
      dist_queues[0].push(point);

//cerr << "s: " << s << "  point: " << point << "  dist: " << dists[ line ][ pixel ] << endl;

    }

    // (2) calculate shortest distances from supplys
    

    
//    int total_scanned = 0;
//    int scanned_count = 0;
    int min_dist = 0;
    int max_dist = 0;
    while(!dist_queues[min_dist].empty()) {  // as long as the labeled_set is not empty, do the following ......
      point = dist_queues[min_dist].front();
      dist_queues[min_dist].pop();

      line = point.y;
      pixel = point.x;

//	if(pixel == 3 && line == 3) cerr << "iter: " << iter << "   min_dist: " << min_dist << "  scanned: " << point << "  dist: " << dists[line][pixel] << endl;

//if(iter == 2) cerr << "min_dist: " << min_dist << "  point: " << point << endl;

      if(visit[line][pixel] == scanned) {
	//if(nodes[line][pixel].supply == demand) scanned_count ++;

	while( dist_queues[min_dist].empty()){
	  min_dist ++;	  
	  if(min_dist >= nr_queues) break;
	}
//	if(pixel == 3 && line == 3) cerr << "after   min_dist: " << min_dist << endl;

	if(min_dist >= nr_queues) break;

	continue;
      }

//      if(nodes[line][pixel].supply == demand) scanned_count ++;
//      total_scanned ++;


      curr_dist = dists[line][pixel];
      visit[line][pixel] = scanned;

      int tmp_mind = infinity;
      // Left pixel ......

      if(pixel > 0 && visit[line][pixel - 1] != scanned) {
//       if(node_flow[line][pixel - 1].toRight > -flow_limit_per_arc) {
	if(node_flow[line][pixel - 1].toRight != 0) reduced_cost = 0;
	else reduced_cost = nodes[line][pixel - 1].rc + pot[line][pixel] - pot[line][pixel - 1];
	d = curr_dist + reduced_cost;
        //uchar cost = nodes[line][pixel - 1].rc;
	//if(node_flow[line][pixel - 1].toRight > 0) d = curr_dist - cost;
        //else d = curr_dist + cost;
	
	if(d < dists[line][pixel - 1]) {
          visit[line][pixel - 1] = labeled;
	  dists[line][pixel - 1] = d;
	  branches[line][pixel - 1] = flow_right;
	  point.x = pixel - 1;
	  point.y = line;
	  dist_queues[d].push(point);
	  if(d > max_dist) max_dist = d;
	  if(d < tmp_mind) tmp_mind = d;

//	if(pixel - 1 == 3 && line == 3) cerr << "iter: " << iter << "  Left: scanned: " << Point(pixel, line) << "  dist: " << dists[line][pixel-1] << endl;
	  // cerr << "LEFT  d : " << d << endl;
	}
//       }
      }
      
      // Right pixel ......
      if(pixel < ncols - 1 && visit[line][pixel + 1] != scanned) {
//       if(node_flow[line][pixel].toRight < flow_limit_per_arc) {
	if(node_flow[line][pixel].toRight != 0) reduced_cost = 0;
	else reduced_cost = nodes[line][pixel].rc + pot[line][pixel] - pot[line][pixel + 1];
	d = curr_dist + reduced_cost;
        //uchar cost = nodes[line][pixel].rc;
	//if(node_flow[line][pixel].toRight < 0) d = curr_dist - cost;
        //else d = curr_dist + cost;

	if(d < dists[line][pixel + 1]) { 	
	  visit[line][pixel + 1] = labeled;
	  dists[line][pixel + 1] = d;
	  branches[line][pixel + 1] = flow_left;
	  point.x = pixel + 1;
	  point.y = line;
	  dist_queues[d].push(point);
	  if(d > max_dist) max_dist = d;
	  if(d < tmp_mind) tmp_mind = d;
	  // cerr << "Right  d : " << d << endl;
//	if(pixel + 1 == 3 && line == 3) cerr << "iter: " << iter << "  Right scanned: " << Point(pixel, line) << "  dist: " << dists[line][pixel+1] << endl;
	}
//       }
      }
      
      // Up pixel ......
      if(line > 0 && visit[line - 1][pixel] != scanned) {
//       if(node_flow[line - 1][pixel].toDown > -flow_limit_per_arc) {
	if(node_flow[line - 1][pixel].toDown != 0) reduced_cost = 0;
	else reduced_cost = nodes[line - 1][pixel].dc + pot[line][pixel] - pot[line - 1][pixel];
	d = curr_dist + reduced_cost;
        //uchar cost = nodes[line - 1][pixel].dc;
	//if(node_flow[line - 1][pixel].toDown > 0) d = curr_dist - cost;
        //else d = curr_dist + cost;
	
	if(d < dists[line - 1][pixel]) {
	  visit[line - 1][pixel] = labeled;
	  dists[line - 1][pixel] = d;
	  branches[line - 1][pixel] = flow_down;
	  point.x = pixel;
	  point.y = line - 1;
	  dist_queues[d].push(point);
	  if(d > max_dist) max_dist = d;
	  if(d < tmp_mind) tmp_mind = d;
	  // cerr << "UP  d : " << d << endl;
//	if(pixel == 3 && line - 1 == 3) cerr << "iter: " << iter << "  UP scanned: " << Point(pixel, line) << "  dist: " << dists[line - 1][pixel] << endl;
	}
//       }
      }
      
      // Down pixel ......
      if(line < nrows - 1 && visit[line + 1][pixel] != scanned) {
//       if(node_flow[line][pixel].toDown < flow_limit_per_arc) {
	if(node_flow[line][pixel].toDown != 0) reduced_cost = 0;
	else reduced_cost = nodes[line][pixel].dc + pot[line][pixel] - pot[line + 1][pixel];
	d = curr_dist + reduced_cost;
        //uchar cost = nodes[line][pixel].dc;
	//if(node_flow[line][pixel].toDown < 0) d = curr_dist - cost;
        //else d = curr_dist + cost;

	if(d < dists[line + 1][pixel]) {
	  visit[line + 1][pixel] = labeled;
	  dists[line + 1][pixel] = d;
	  branches[line + 1][pixel] = flow_up;
	  point.x = pixel;
	  point.y = line + 1;
	  dist_queues[d].push(point);
	  if(d < tmp_mind) tmp_mind = d;
	  if(d > max_dist) max_dist = d;

//	if(pixel == 3 && line + 1 == 3) cerr << "iter: " << iter << "  Down scanned: " << Point(pixel, line) << "  dist: " << dists[line + 1][pixel] << endl;
	  // cerr << "Down  d : " << d << endl;
	}
	//if(d < tmp_mind) tmp_mind = d;
//       }
      }
      
      if(tmp_mind < min_dist) min_dist = tmp_mind;
      
      while( dist_queues[min_dist].empty()){
	min_dist ++;
	if(min_dist > max_dist) break;
      }
      if(min_dist > max_dist) break;
      //if(iter > 0 && scanned_count == supplys_left) cerr << "total_scanned: " << total_scanned << endl;
      //if(iter > 0 && scanned_count == supplys_left) break;
    }
    
/*
    int start_dist = 0;
    if(min_dist > 0) start_dist = min_dist;
    for(int i = start_dist; i < nr_queues; i++) {
      dist_queues[i] = queue<Point>();
    }
*/
    
/*
    // check the distances ......
      cerr << "Check dist ...... \n";
    for(int line = 0; line < nrows; line ++) {
      cerr << "line : ";
      for(int pixel = 0; pixel < ncols; pixel ++) {
	cerr << dists[line][pixel] << " ";
	//if(dists[line][pixel] != infinity) cerr << dists[line][pixel] << " ";
	//else cerr << dists[line][pixel] << " inf ";
      }
      cerr << endl;
    }
    cerr << endl;
*/
    
   
//    if(L0L1_mode == 1) {
      for(int line = 0; line < nrows; line ++) {
        for(int pixel = 0; pixel < ncols; pixel ++) {
//	  if(dists[line][pixel] > 0 && visit[line][pixel] == scanned) {
	    pot[line][pixel] += dists[line][pixel];
//	  }	  
        }
      }
//    } 

    
/*    
    // check the PIs ......
      cerr << "Check potential ......\n ";
    for(int line = 0; line < nrows; line ++) {
      cerr << "pot: ";
      for(int pixel = 0; pixel < ncols; pixel ++) {
	cerr << pot[line][pixel] << "  ";
      }
      cerr << endl;
    }
    cerr << endl;
*/    
     
  //  exit(0);
    
    
    cerr << "iter: " << iter << "  supplys_left: " << supplys_left << endl;
//    cerr << "iter: " << iter << "  supplys_left: " << supplys_left << "  scanned_count: "  << scanned_count << "  total_scanned: " << total_scanned << endl;
//    cerr << "iter: " << iter << "  supplys_left: " << supplys_left << "  min_dist: " << min_dist << "  max_dist: " << max_dist << "  nr_queues: " << nr_queues << endl; //"  scanned_count: "  << scanned_count << endl;

    int count = 0;
    for(int i = 0; i < demands; i++) {
      int line = T[i].y;
      int pixel = T[i].x;
      if(nodes[line][pixel].supply == 0) continue;
      indexes[count] = i;
      //dd[count] = count;
            dd[count] = pot[line][pixel];
      //dd[count] = dists[line][pixel];
//cerr << "count: " << count << "  point: " << Point(pixel, line) << "  supply: " << (int)nodes[line][pixel].supply << "  dd: " << dd[count] << endl;
      count ++;
    }
    heapSort (count, dd, indexes);
    
//cerr << "count: " << count << endl;
//cerr << "end of heapSort() !!!!!! \n";

    for(int i = 0; i < count; i++) {
      int index = indexes[i];
      int line = T[index].y;
      int pixel = T[index].x;
      if(nodes[line][pixel].supply == 0) continue;
      flow.flowdir.clear();
      
      flow.xstart = pixel;
      flow.ystart = line;
      

      int branch_taken = 0;

      //int integrated_cost = 0;

//cerr << "add the branches ...... \n";
//cerr << Point(pixel, line) << "  ";

      do {
	flow.flowdir.push_back(branches[line][pixel]);
	
	if(branches[line][pixel] == flow_up) {
	  //if(pixel != 0 && pixel != ncols - 1 && new_flow[line - 1][pixel].toDown != 0) {
	  if(new_flow[line - 1][pixel].toDown != 0) {
	    branch_taken = 1;
	    break;
	  }

	  line --;
	}
	else if(branches[line][pixel] == flow_down) {
	  //if(pixel != 0 && pixel != ncols - 1 && new_flow[line][pixel].toDown != 0) {
	  if(new_flow[line][pixel].toDown != 0) {
	    branch_taken = 1;
	    break;
	  }

	  line ++;
	}
	else if(branches[line][pixel] == flow_left) {
	  //if(line != 0 && line != nrows - 1 && new_flow[line][pixel - 1].toRight != 0) {
	  if(new_flow[line][pixel - 1].toRight != 0) {
	    branch_taken = 1;
	    break;
	  }

	  pixel --;
	}
	else if(branches[line][pixel] == flow_right) {
	  //if(line != 0 && line != nrows - 1 && new_flow[line][pixel].toRight != 0) {
	  if(new_flow[line][pixel].toRight != 0) {
	    branch_taken = 1;
	    break;
	  }

	  pixel ++;
	}

//cerr << Point(pixel, line) << "  ";

      } while(branches[line][pixel] != noflow);
      
//cerr << endl;

//     cerr << "for flow start at: " << flow.xstart << ", " << flow.ystart  << "  branch_taken: " << branch_taken << "  End at supply " << pixel << " " << line << "  supply: " << (int)nodes[line][pixel].supply << endl;

      if(branch_taken == 1) continue;
      // if(integrated_cost != 0) continue;
      
      if(nodes[line][pixel].supply == 0) continue;  // if the supply node is balanced terminate !!
      supplys_left --;
      nodes[line][pixel].supply -= supply;                  // supply
      nodes[flow.ystart][flow.xstart].supply -= demand;     // demand  

//cerr << "   start supply: " << (int)nodes[flow.ystart][flow.xstart].supply << "  end supply: " << (int)nodes[line][pixel].supply << endl;

      // update the new flows ......
      
      line = flow.ystart;
      pixel = flow.xstart;
      for(int j = 0; j < flow.flowdir.size(); j ++) {
	if(flow.flowdir[j] == flow_up) {
	  if(pixel > 0 && pixel < ncols - 1) new_flow[line - 1][pixel].toDown ++;
	  line --;
	}
	else if(flow.flowdir[j] == flow_down) {
	  if(pixel > 0 && pixel < ncols - 1) new_flow[line][pixel].toDown --;
	  line ++;
	}
	else if(flow.flowdir[j] == flow_left) {
	  if(line > 0 && line < nrows - 1) new_flow[line][pixel - 1].toRight ++;
	  pixel --;
	}
	else if(flow.flowdir[j] == flow_right) {
	  if(line > 0 && line < nrows - 1) new_flow[line][pixel].toRight --;
	  pixel ++;
	}
      }  // for loop 
      
      
    } // for(int i
    
//cerr << "update node_flow[][] ...... \n";    

    // update the node flow ......

//    int max_flows_per_arc = 0; 
//    Point max_flows_location;   

    for(int line = 0; line < nrows; line ++) {
      for(int pixel = 0; pixel < ncols; pixel ++) {
    	if(new_flow[line][pixel].toRight != 0) {
	  node_flow[line][pixel].toRight += new_flow[line][pixel].toRight;
	  new_flow[line][pixel].toRight = 0;
	}
    	if(new_flow[line][pixel].toDown != 0) {
	  node_flow[line][pixel].toDown += new_flow[line][pixel].toDown;
	  new_flow[line][pixel].toDown = 0;
	}
	/*
	if(abs(new_flow[line][pixel].toRight) > 1 || abs(new_flow[line][pixel].toDown) > 1) 
	  cerr << " New Flow  line: " << line << "  pixel: " << pixel << "  To Right: " << abs(new_flow[line][pixel].toRight) << "  To Down : " << abs(new_flow[line][pixel].toDown) << endl;
	if(abs(node_flow[line][pixel].toRight) > 1 || abs(node_flow[line][pixel].toDown) > 1) 
	  cerr << " total Flow line: " << line << "  pixel: " << pixel << "  ToRight: " << abs(node_flow[line][pixel].toRight) << "  To Down: " << abs(node_flow[line][pixel].toDown)<< endl;
	*/
/*
        if(abs((int)node_flow[line][pixel].toRight) > max_flows_per_arc) {
	  max_flows_per_arc = abs((int)node_flow[line][pixel].toRight);
	  max_flows_location.set(pixel, line);
	}
        if(abs((int)node_flow[line][pixel].toDown) > max_flows_per_arc) {
	  max_flows_per_arc = abs((int)node_flow[line][pixel].toDown);
	  max_flows_location.set(pixel, line);
	}
*/
      }
    }

      

//    cerr << "iter: " << iter << "  supplys_left: " << supplys_left << "  max_flows_per_arc: " << max_flows_per_arc << "  max_flows_location: " << max_flows_location << endl;
//    cerr << "(0, 499) toRight: " << (int)node_flow[499][0].toRight << "  toDown: " << (int)node_flow[499][0].toDown << endl;
    iter++;  
  }

/*
  cerr << "toRight flow ..... \n";
  for(int line = 0; line < nrows; line ++) {
    for(int pixel = 0; pixel < ncols; pixel ++) {
      cerr << (int) node_flow[line][pixel].toRight << " ";
    }
    cerr << endl;
  }

  cerr << "toDown flow ...... \n";
  for(int line = 0; line < nrows; line ++) {
    for(int pixel = 0; pixel < ncols; pixel ++) {
      cerr << (int) node_flow[line][pixel].toDown << " ";
    }
    cerr << endl;
  }
*/

  delete[] indexes;
  delete[] dd;

  if(dist_queues) delete[] dist_queues;
  delete[] S;
  delete[] T;

  delete branch_patch;
  delete visit_patch;
  delete dist_patch;
  delete pot_patch;

  delete new_flow_patch;

  // cerr << "Before return the node_flow_patch !!!!! \n";
/*
      cerr << "to-right flows ......\n";
      for(int i = 0; i < nrows; i++) {
 	for(int j = 0; j < ncols; j++) {
	  cerr << (int)node_flow[i][j].toRight << " ";
	}
	cerr << endl;
      }
      cerr << "to-down flows ......\n";
      for(int i = 0; i < nrows; i++) {
 	for(int j = 0; j < ncols; j++) {
	  cerr << (int)node_flow[i][j].toDown << " ";
	}
	cerr << endl;
      }
      cerr << endl;

  */

  return node_flow_patch;
}

