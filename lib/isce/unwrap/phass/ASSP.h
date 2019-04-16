// Copyright (c) 2017-, California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged.
// All rights reserved.
// 
// Author(s):
// 
//  ----------------------------------------------------------------------------
//  Author:  Xiaoqing Wu
// 

#ifndef ASSP_H
#define ASSP_H

#include "Seed.h"
#include "Point.h"
#include "constants.h"
#include "DataPatch.h"
#include <complex>
#include <list>
#include <set>
#include <queue>
#include <vector>

using namespace std;

typedef unsigned char  uchar;
typedef unsigned short ushort;
typedef unsigned int   uint;

#define cost_scale      100
#define no_data_value   -10000.0

#define noflow      0
#define flow_up     1
#define flow_down   2
#define flow_left   3
#define flow_right  4
  
#define edge_up     0x01
#define edge_down   0x02
#define edge_left   0x04
#define edge_right  0x08
#define edge_all    0x0F

  
#define flow_limit_per_arc 4

struct Node {
//  uchar edge_flag;     // First 4 bits for edge connections. The rest 4 bits reserved.
  char  supply;      // 1: supply;  -1: demand; 0: neutral.
//  ushort rc;          // right (pixel + 1) cost
//  ushort dc;          // down (line + 1) cost
  uchar rc;          // right (pixel + 1) cost
  uchar dc;          // down (line + 1) cost
};



struct Flow {
  //ushort xstart;
  //ushort ystart;
  int xstart;
  int ystart;
  vector<uchar> flowdir; // flow directions
};
  
struct NodeFlow {
  char toRight;
  char toDown;
};


// vector<Flow> solve_assp(DataPatch<Node> *node_patch);

DataPatch<NodeFlow> *solve(DataPatch<Node> *node_patch);

DataPatch<Node> *make_node_patch(int nr_lines, int nr_pixels, float **corr_data, float **phase_data, double qthresh = 0);
DataPatch<Node> *make_node_patch(DataPatch<fcomplex> *int_patch, double qthresh = 0);
void make_node_patch(char *int_file, char *corr_file, char *amp_file, char *layover_file,
		     int start_line, int nr_lines, int nr_pixels, double corr_th, double phase_th, double amp_th,
		     DataPatch<Node> **node_patch, DataPatch<float> **phase_patch, double minimum_corr = 0.0,
			double good_corr = 0.5, double const_phase_to_remove = 0.0, int sq = 1);


void create_seeds(DataPatch<NodeFlow> *flows_patch, int minimum_nr_pixels, int& nr_seeds, Seed **seeds); // seeds only
DataPatch<char>* unwrap_adjust_seeds(DataPatch<NodeFlow> *flows_patch, float **phase_data, int nr_seeds, Seed *seeds); 
DataPatch<char>* unwrap_assp(DataPatch<NodeFlow> *flows_patch, float **phase_data, int nr_seeds, Seed *seeds);


//void flood_fill(int line, int pixel, vector<USPoint>& workv, queue<USPoint>& workq, int nr_lines, int nr_pixels, int **region_map, Node **nodes, 
//		char **visit, uchar curr_weight_th, uchar lower_cost_th);
//void flood_fill(int line, int pixel, queue<USPoint>& workq, int nr_lines, int nr_pixels, int **region_map, char **visit);
//void flood_fill_residues(int line, int pixel, queue<USPoint>& workq, int nr_lines, int nr_pixels, int **region_map, char **visit);

DataPatch<int> * generate_regions(DataPatch<NodeFlow> *flows_patch, int nr_seeds, Seed *seeds);
void generate_regions(DataPatch<NodeFlow> *flows_patch, int nr_seeds, Seed *seeds, int **region_map);

#endif
