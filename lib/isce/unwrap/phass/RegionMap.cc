// Copyright (c) 2017-, California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged.
// All rights reserved.
// 
// Author(s):
// 
//  ======================================================================
// 
//  FILENAME: RegionMap.cc
//   
//  CREATED BY: Xiaoqing WU
// 
//  ======================================================================


#include "RegionMap.h"
#include "DataPatch.h"

int create_region_map(int lines, int pixels, bool **mask, float **normalized_heights, int **region_map, int min_pixels_per_region)
{
  DataPatch<bool> *visit_patch = new DataPatch<bool>(pixels, lines);
  bool **visit = visit_patch->get_data_lines_ptr();

  for(int line = 0; line < lines; line ++) {
    for(int pixel = 0; pixel < pixels; pixel ++) {
      visit[line][pixel] = false;
      region_map[line][pixel] = -1;
    }
  }
  
  queue<Point> workq;

  int region_id = 0;
  double th = 0.45;
  
  for(int ii = 0; ii < lines; ii ++) {
    for(int jj = 0; jj < pixels; jj ++) {
      if(mask[ii][jj] == false) continue;
      if(visit[ii][jj]) continue;

      workq.push(Point(jj, ii));

      region_map[ii][jj] = region_id;
      visit[ii][jj] = true;

      while( !workq.empty() ) {
        Point point = workq.front();
        workq.pop();
        int line  = point.get_Y();
        int pixel = point.get_X();
        float hgt = normalized_heights[line][pixel];

        if(line > 0) {
	  double diff = fabs(normalized_heights[line-1][pixel] - hgt);
	  if( !visit[line - 1][pixel] && mask[line - 1][pixel] && diff < th) { 
	    workq.push(Point(pixel, line - 1));
	    visit[line - 1][pixel] = true;
	    region_map[line - 1][pixel] = region_id;
	  }
	}
        if(line < lines - 1) {
	  double diff = fabs(normalized_heights[line+1][pixel] - hgt);
	  if( !visit[line + 1][pixel] && mask[line + 1][pixel] && diff < th) { 
	    workq.push(Point(pixel, line + 1));
	    visit[line + 1][pixel] = true;
	    region_map[line + 1][pixel] = region_id;
	  }
	}
        if(pixel > 0 ) {
	  double diff = fabs(normalized_heights[line][pixel-1] - hgt);
	  if( !visit[line][pixel - 1] && mask[line][pixel - 1] && diff < th) { 
	    workq.push(Point(pixel - 1, line));
	    visit[line][pixel - 1] = true;
	    region_map[line][pixel - 1] = region_id;
	  }
	}
        if(pixel < pixels - 1 ) {
	  double diff = fabs(normalized_heights[line][pixel+1] - hgt);
	  if( !visit[line][pixel + 1] && mask[line][pixel + 1] && diff < th) { 
	    workq.push(Point(pixel + 1, line));
	    visit[line][pixel + 1] = true;
	    region_map[line][pixel + 1] = region_id;
	  }
	}
	
        if(line > 0 && pixel > 0) {
	  double diff = fabs(normalized_heights[line-1][pixel-1] - hgt);
	  if( !visit[line - 1][pixel-1] && mask[line - 1][pixel-1] && diff < th) { 
	    workq.push(Point(pixel-1, line - 1));
	    visit[line - 1][pixel-1] = true;
	    region_map[line - 1][pixel-1] = region_id;
	  }
	}
        if(line > 0 && pixel < pixels - 1) {
	  double diff = fabs(normalized_heights[line-1][pixel+1] - hgt);
	  if( !visit[line - 1][pixel+1] && mask[line - 1][pixel+1] && diff < th) { 
	    workq.push(Point(pixel+1, line - 1));
	    visit[line - 1][pixel+1] = true;
	    region_map[line - 1][pixel+1] = region_id;
	  }
	}
        if(line < lines - 1 && pixel > 0) {
	  double diff = fabs(normalized_heights[line+1][pixel-1] - hgt);
	  if( !visit[line + 1][pixel-1] && mask[line + 1][pixel-1] && diff < th) { 
	    workq.push(Point(pixel-1, line + 1));
	    visit[line + 1][pixel-1] = true;
	    region_map[line + 1][pixel-1] = region_id;
	  }
	}
        if(line < lines - 1 && pixel < pixels - 1) {
	  double diff = fabs(normalized_heights[line+1][pixel+1] - hgt);
	  if( !visit[line + 1][pixel+1] && mask[line + 1][pixel+1] && diff < th) { 
	    workq.push(Point(pixel+1, line + 1));
	    visit[line + 1][pixel+1] = true;
	    region_map[line + 1][pixel+1] = region_id;
	  }
	}

      }
      region_id ++;
    }
  }

  delete visit_patch;

  list<USPoint> *point_list = make_point_list(lines, pixels, region_map, region_id);
  for(int i = 0; i < lines; i++) {
    for(int j = 0; j < pixels; j++) {
      region_map[i][j] = -1;
    }
  }

  int nr_regions = region_id;

//  cerr << "nr_regions: " << nr_regions << endl;

  region_id = 0;
  for(int id = 0; id < nr_regions; id++) {

//cerr << "id: " << id << "  size: " << point_list[id].size() << "  start: " << *(point_list[id].begin()) << endl;

    if(point_list[id].size() < min_pixels_per_region) continue;
    for(list<USPoint>::iterator it = point_list[id].begin(); it != point_list[id].end(); it++) {
      USPoint point = *it;
      int line = point.y;
      int pixel = point.x;
      region_map[line][pixel] = region_id;
    }    
    region_id ++;
  }  
  
  for(int id = 0; id < nr_regions; id++) {
    point_list[id].clear();
  }
  delete[] point_list;
  
//  exit(0);

  return region_id;
}


int create_region_map(int lines, int pixels, bool **mask, unsigned char **disc_data,
                       int **region_map, int min_pixels_per_region)
// return the number of regions and assign each region with a region id and -1 to no-mask pixels 
// Inputs: lines, pixels, mask, disc_data, region_map
// Output: region_map, nr_regions, region_points
{
  DataPatch<bool> *visit_patch = new DataPatch<bool>(pixels, lines);
  bool **visit = visit_patch->get_data_lines_ptr();

  for(int line = 0; line < lines; line ++) {
    for(int pixel = 0; pixel < pixels; pixel ++) {
      visit[line][pixel] = false;
      region_map[line][pixel] = -1;
    }
  }
  
  queue<Point> workq;

  int region_id = 0;
  
  for(int ii = 0; ii < lines; ii ++) {
    for(int jj = 0; jj < pixels; jj ++) {
      if(mask[ii][jj] == false) continue;
      if(visit[ii][jj]) continue;

      workq.push(Point(jj, ii));

      region_map[ii][jj] = region_id;
      visit[ii][jj] = true;

      while( !workq.empty() ) {
        Point point = workq.front();
        workq.pop();
        int line  = point.get_Y();
        int pixel = point.get_X();

//if(line == 1365 && pixel == 1924) cerr << "region_id: " << region_id << "  point: " << point << endl;
//if(region_id == 1151) cerr << "region_id: " << region_id << "  point: " << point << endl;

        if(line > 0 && (disc_data[line][pixel] & face_up) == 0) {
//if(line == 1367 && pixel == 1931) cerr << "  up   " << Point(pixel, line - 1) << "  size: " << workq.size() << "  visit: " << (int)visit[line - 1][pixel] << "  mask: " << (int)mask[line - 1][pixel] << "  disc: " << (int)disc_data[line - 1][pixel] << endl;
	  if( !visit[line - 1][pixel] && mask[line - 1][pixel] ) { 
	    workq.push(Point(pixel, line - 1));
	    visit[line - 1][pixel] = true;
	    region_map[line - 1][pixel] = region_id;

if(line == 1359 && pixel == 1924) cerr << "  up   " << Point(pixel, line - 1) << "  size: " << workq.size() << "  visit: " << (int)visit[line - 1][pixel] << "  mask: " << (int)mask[line - 1][pixel] << "  disc: " << (int)disc_data[line - 1][pixel] << endl;
	  }
	}
        if(line < lines - 1 && (disc_data[line][pixel] & face_down) == 0) {
	  if( !visit[line + 1][pixel] && mask[line + 1][pixel] ) { 
	    workq.push(Point(pixel, line + 1));
	    visit[line + 1][pixel] = true;
	    region_map[line + 1][pixel] = region_id;;

//cerr << "     " << Point(pixel, line + 1) << "  size: " << workq.size() << endl;
	  }
	}
        if(pixel > 0 && (disc_data[line][pixel] & face_left) == 0) {
	  if( !visit[line][pixel - 1] && mask[line][pixel - 1] ) { 
	    workq.push(Point(pixel - 1, line));
	    visit[line][pixel - 1] = true;
	    region_map[line][pixel - 1] = region_id;

if(line == 1359 && pixel == 1924) cerr << "   left  " << Point(pixel - 1, line) << "  size: " << workq.size() << "  visit: " << (int)visit[line][pixel - 1] << "  mask: " << (int)mask[line][pixel - 1] << "  disc: " << (int)disc_data[line][pixel - 1] << endl;

//cerr << "     " << Point(pixel - 1, line) << "  size: " << workq.size() << endl;
	  }
	}
        if(pixel < pixels - 1 && (disc_data[line][pixel] & face_right) == 0) {
	  if( !visit[line][pixel + 1] && mask[line][pixel + 1] ) { 
	    workq.push(Point(pixel + 1, line));
	    visit[line][pixel + 1] = true;
	    region_map[line][pixel + 1] = region_id;

if(line == 1359 && pixel == 1924) cerr << "   right  " << Point(pixel + 1, line) << "  size: " << workq.size() << "  visit: " << (int)visit[line][pixel + 1] << "  mask: " << (int)mask[line][pixel + 1] << "  disc: " << (int)disc_data[line][pixel + 1] << endl;

//cerr << "     " << Point(pixel - 1, line) << "  size: " << workq.size() << endl;
	  }
	}
	
      }
      region_id ++;
    }
  }

  delete visit_patch;

  list<USPoint> *point_list = make_point_list(lines, pixels, region_map, region_id);
  for(int i = 0; i < lines; i++) {
    for(int j = 0; j < pixels; j++) {
      region_map[i][j] = -1;
    }
  }

  int nr_regions = region_id;

//  cerr << "nr_regions: " << nr_regions << endl;

  region_id = 0;
  for(int id = 0; id < nr_regions; id++) {

//cerr << "id: " << id << "  size: " << point_list[id].size() << "  start: " << *(point_list[id].begin()) << endl;

    if(point_list[id].size() < min_pixels_per_region) continue;
    for(list<USPoint>::iterator it = point_list[id].begin(); it != point_list[id].end(); it++) {
      USPoint point = *it;
      int line = point.y;
      int pixel = point.x;
      region_map[line][pixel] = region_id;
    }    
    region_id ++;
  }  
  
  for(int id = 0; id < nr_regions; id++) {
    point_list[id].clear();
  }
  delete[] point_list;
  
//  exit(0);

  return region_id;
}


int create_region_map(int lines, int pixels, float **unw_phases, int **region_map, int min_pixels_per_region, float nodata)
{
  DataPatch<int> *in_patch = new DataPatch<int>(pixels, lines);
  int **in_region_map = in_patch->get_data_lines_ptr();
  for(int line = 0; line < lines; line ++) {
    for(int pixel = 0; pixel < pixels; pixel ++) {
      in_region_map[line][pixel] = region_map[line][pixel];
      region_map[line][pixel] = -1;
    }
  }

  int nr_lines = lines;
  int nr_pixels = pixels;
  float no_phase_value = nodata;

    DataPatch<bool> *visit_patch = new DataPatch<bool>(nr_pixels, nr_lines);
    bool **visit = visit_patch->get_data_lines_ptr();
    for(int line = 0; line < nr_lines; line ++ ) {
      for(int pixel = 0; pixel < nr_pixels; pixel ++) {
        visit[line][pixel] = false;
      }
    }

    int rid = 0;
    queue<Point> workq;
    for(int ii = 0; ii < nr_lines; ii ++ ) {
      for(int jj = 0; jj < nr_pixels; jj ++) {
        if(unw_phases[ii][jj] == no_phase_value) continue;
        if(visit[ii][jj]) continue;
        workq.push(Point(jj, ii));
        int this_rid = in_region_map[ii][jj];

        while( !workq.empty() ) {
          Point point = workq.front();
          workq.pop();
          int line  = point.get_Y();
          int pixel = point.get_X();

	  visit[line][pixel] = true;
	  region_map[line][pixel] = rid;
          int line_plus = line + 1;
          int line_minus = line - 1;
          int pixel_plus = pixel + 1;
          int pixel_minus = pixel - 1;

          if(line > 0) {              // facing up ......
            if(unw_phases[line_minus][pixel] != no_phase_value && visit[line_minus][pixel] == false && in_region_map[line_minus][pixel] == this_rid) {
	      workq.push(Point(pixel, line_minus));
	      region_map[line_minus][pixel] = rid;
	      visit[line_minus][pixel] = true;
            }	
          }
          if(line < nr_lines - 1) {   // facing down ...... 
            if(unw_phases[line_plus][pixel] != no_phase_value && visit[line_plus][pixel] == false && in_region_map[line_plus][pixel] == this_rid) {
	      workq.push(Point(pixel, line_plus));
	      region_map[line_plus][pixel] = rid;
	      visit[line_plus][pixel] = true;
            }
          }
          if(pixel > 0) {             // facing left ......
            if(unw_phases[line][pixel_minus] != no_phase_value && visit[line][pixel_minus] == false && in_region_map[line][pixel_minus] == this_rid) {
	      workq.push(Point(pixel_minus, line));
	      region_map[line][pixel_minus] = rid;
	      visit[line][pixel_minus] = true;
            }	
          }
          if(pixel < nr_pixels - 1) {// facing right ......           // facing left ......
            if(unw_phases[line][pixel_plus] != no_phase_value && visit[line][pixel_plus] == false && in_region_map[line][pixel_plus] == this_rid) {
	      workq.push(Point(pixel_plus, line));
	      region_map[line][pixel_plus] = rid;
	      visit[line][pixel_plus] = true;
            }	
          }
        }
        rid ++;
      }
    }
    delete visit_patch;
   
    delete in_patch;

  return rid;
}

int create_region_map(int lines, int pixels, bool **mask, int **region_map, int min_pixels_per_region)
// return the number of regions and assign each region with a region id and -1 to no-mask pixels 
// Inputs: lines, pixels, mask, region_map
// Output: region_map, nr_regions, region_points

{
  DataPatch<bool> *visit_patch = new DataPatch<bool>(pixels, lines);
  bool **visit = visit_patch->get_data_lines_ptr();

  for(int line = 0; line < lines; line ++) {
    for(int pixel = 0; pixel < pixels; pixel ++) {
      visit[line][pixel] = false;
      region_map[line][pixel] = -1;
    }
  }
  
  queue<Point> workq;

  int region_id = 0;
  
  for(int ii = 0; ii < lines; ii ++) {
    for(int jj = 0; jj < pixels; jj ++) {
      if(mask[ii][jj] == false) continue;
      if(visit[ii][jj]) continue;

      workq.push(Point(jj, ii));

      region_map[ii][jj] = region_id;
      visit[ii][jj] = true;

      while( !workq.empty() ) {
        Point point = workq.front();
        workq.pop();
        int line  = point.get_Y();
        int pixel = point.get_X();

        if(line > 0) {
	  if( !visit[line - 1][pixel] && mask[line - 1][pixel]) {
	    workq.push(Point(pixel, line - 1));
	    visit[line - 1][pixel] = true;
	    region_map[line - 1][pixel] = region_id;
//cerr << "     " << Point(pixel, line - 1) << "  size: " << workq.size() << endl;

	  }
	}
        if(line < lines - 1) {
	  if( !visit[line + 1][pixel] && mask[line + 1][pixel]) {
	    workq.push(Point(pixel, line + 1));
	    visit[line + 1][pixel] = true;
	    region_map[line + 1][pixel] = region_id;
//cerr << "     " << Point(pixel, line + 1) << "  size: " << workq.size() << endl;
	  }
	}
        if(pixel > 0) {
	  if( !visit[line][pixel - 1] && mask[line][pixel - 1] ) {
	    workq.push(Point(pixel - 1, line));
	    visit[line][pixel - 1] = true;
	    region_map[line][pixel - 1] = region_id;
//cerr << "     " << Point(pixel - 1, line) << "  size: " << workq.size() << endl;
	  }
	}
        if(pixel < pixels - 1) {
	  if( !visit[line][pixel + 1] && mask[line][pixel + 1] ) {
	    workq.push(Point(pixel + 1, line));
	    visit[line][pixel + 1] = true;
	    region_map[line][pixel + 1] = region_id;
//cerr << "     " << Point(pixel - 1, line) << "  size: " << workq.size() << endl;
	  }
	}
	
      }
      region_id ++;
    }
  }

  delete visit_patch;

  list<USPoint> *point_list = make_point_list(lines, pixels, region_map, region_id);
  for(int i = 0; i < lines; i++) {
    for(int j = 0; j < pixels; j++) {
      region_map[i][j] = -1;
    }
  }

  int nr_regions = region_id;

//  cerr << "nr_regions: " << nr_regions << endl;

  region_id = 0;
  for(int id = 0; id < nr_regions; id++) {

//cerr << "id: " << id << "  size: " << point_list[id].size() << "  start: " << *(point_list[id].begin()) << endl;

    if(point_list[id].size() < min_pixels_per_region) continue;
    for(list<USPoint>::iterator it = point_list[id].begin(); it != point_list[id].end(); it++) {
      USPoint point = *it;
      int line = point.y;
      int pixel = point.x;
      region_map[line][pixel] = region_id;
    }    
    region_id ++;
  }  
  
  for(int id = 0; id < nr_regions; id++) {
    point_list[id].clear();
  }
  delete[] point_list;
  
//  exit(0);

  return region_id;
}

void make_point_list(int lines, int pixels, int **region_map, int &nr_regions, list<USPoint> **point_list)
{
  set<int> region_set;
  for(int line = 0; line < lines; line ++) {
    for(int pixel = 0; pixel < pixels; pixel ++) {
      int region_id = region_map[line][pixel];
      region_set.insert(region_id);
    }
  }
  nr_regions = region_set.size();
  region_set.clear();

  list<USPoint> *ret_point_list = new list<USPoint>[nr_regions];
  for(int line = 0; line < lines; line ++) {
    for(int pixel = 0; pixel < pixels; pixel ++) {
      int region_id = region_map[line][pixel];
      if(region_id >= 0 && region_id < nr_regions) {
	ret_point_list[region_id].push_back(USPoint(pixel, line));
      }
    }
  }
  *point_list = ret_point_list;
}

list<USPoint> *make_point_list(int lines, int pixels, int **region_map, int nr_regions)
{
  list<USPoint> *point_list = new list<USPoint>[nr_regions];
  for(int line = 0; line < lines; line ++) {
    for(int pixel = 0; pixel < pixels; pixel ++) {
      int region_id = region_map[line][pixel];
      if(region_id >= 0 && region_id < nr_regions) {
	point_list[region_id].push_back(USPoint(pixel, line));
      }
    }
  }
  return point_list;
}


int create_region_map(int lines, int pixels, int **input_regions, float **normalized_heights, int **region_map, int min_pixels_per_region)
{
  DataPatch<bool> *visit_patch = new DataPatch<bool>(pixels, lines);
  bool **visit = visit_patch->get_data_lines_ptr();

  for(int line = 0; line < lines; line ++) {
    for(int pixel = 0; pixel < pixels; pixel ++) {
      visit[line][pixel] = false;
      region_map[line][pixel] = -1;
    }
  }
  
  queue<Point> workq;

  int region_id = 0;
  double th = 0.45;
  
  for(int ii = 0; ii < lines; ii ++) {
    for(int jj = 0; jj < pixels; jj ++) {
      if(input_regions[ii][jj] == -1) continue;
      if(visit[ii][jj]) continue;

      workq.push(Point(jj, ii));

      region_map[ii][jj] = region_id;
      visit[ii][jj] = true;

      int original_id = input_regions[ii][jj];

      while( !workq.empty() ) {
        Point point = workq.front();
        workq.pop();
        int line  = point.get_Y();
        int pixel = point.get_X();
        float hgt = normalized_heights[line][pixel];

        if(line > 0) {
	  double diff = fabs(normalized_heights[line-1][pixel] - hgt);
	  if( !visit[line - 1][pixel] && input_regions[line - 1][pixel] == original_id && diff < th) { 
	    workq.push(Point(pixel, line - 1));
	    visit[line - 1][pixel] = true;
	    region_map[line - 1][pixel] = region_id;
	  }
	}
        if(line < lines - 1) {
	  double diff = fabs(normalized_heights[line+1][pixel] - hgt);
	  if( !visit[line + 1][pixel] && input_regions[line + 1][pixel] == original_id && diff < th) { 
	    workq.push(Point(pixel, line + 1));
	    visit[line + 1][pixel] = true;
	    region_map[line + 1][pixel] = region_id;
	  }
	}
        if(pixel > 0 ) {
	  double diff = fabs(normalized_heights[line][pixel-1] - hgt);
	  if( !visit[line][pixel - 1] && input_regions[line][pixel - 1] == original_id && diff < th) { 
	    workq.push(Point(pixel - 1, line));
	    visit[line][pixel - 1] = true;
	    region_map[line][pixel - 1] = region_id;
	  }
	}
        if(pixel < pixels - 1 ) {
	  double diff = fabs(normalized_heights[line][pixel+1] - hgt);
	  if( !visit[line][pixel + 1] && input_regions[line][pixel + 1] == original_id && diff < th) { 
	    workq.push(Point(pixel + 1, line));
	    visit[line][pixel + 1] = true;
	    region_map[line][pixel + 1] = region_id;
	  }
	}
	
        if(line > 0 && pixel > 0) {
	  double diff = fabs(normalized_heights[line-1][pixel-1] - hgt);
	  if( !visit[line - 1][pixel-1] && input_regions[line - 1][pixel-1] == original_id && diff < th) { 
	    workq.push(Point(pixel-1, line - 1));
	    visit[line - 1][pixel-1] = true;
	    region_map[line - 1][pixel-1] = region_id;
	  }
	}
        if(line > 0 && pixel < pixels - 1) {
	  double diff = fabs(normalized_heights[line-1][pixel+1] - hgt);
	  if( !visit[line - 1][pixel+1] && input_regions[line - 1][pixel+1] == original_id && diff < th) { 
	    workq.push(Point(pixel+1, line - 1));
	    visit[line - 1][pixel+1] = true;
	    region_map[line - 1][pixel+1] = region_id;
	  }
	}
        if(line < lines - 1 && pixel > 0) {
	  double diff = fabs(normalized_heights[line+1][pixel-1] - hgt);
	  if( !visit[line + 1][pixel-1] && input_regions[line + 1][pixel-1] == original_id && diff < th) { 
	    workq.push(Point(pixel-1, line + 1));
	    visit[line + 1][pixel-1] = true;
	    region_map[line + 1][pixel-1] = region_id;
	  }
	}
        if(line < lines - 1 && pixel < pixels - 1) {
	  double diff = fabs(normalized_heights[line+1][pixel+1] - hgt);
	  if( !visit[line + 1][pixel+1] && input_regions[line + 1][pixel+1] == original_id && diff < th) { 
	    workq.push(Point(pixel+1, line + 1));
	    visit[line + 1][pixel+1] = true;
	    region_map[line + 1][pixel+1] = region_id;
	  }
	}

      }
      region_id ++;
    }
  }

  delete visit_patch;

  list<USPoint> *point_list = make_point_list(lines, pixels, region_map, region_id);
  for(int i = 0; i < lines; i++) {
    for(int j = 0; j < pixels; j++) {
      region_map[i][j] = -1;
    }
  }

  int nr_regions = region_id;

//  cerr << "nr_regions: " << nr_regions << endl;

  region_id = 0;
  for(int id = 0; id < nr_regions; id++) {

//cerr << "id: " << id << "  size: " << point_list[id].size() << "  start: " << *(point_list[id].begin()) << endl;

    if(point_list[id].size() < min_pixels_per_region) continue;
    for(list<USPoint>::iterator it = point_list[id].begin(); it != point_list[id].end(); it++) {
      USPoint point = *it;
      int line = point.y;
      int pixel = point.x;
      region_map[line][pixel] = region_id;
    }    
    region_id ++;
  }  
  
  for(int id = 0; id < nr_regions; id++) {
    point_list[id].clear();
  }
  delete[] point_list;
  
//  exit(0);

  return region_id;
}


int create_region_map(int lines, int pixels, int **input_regions, int **region_map, int min_pixels_per_region)
{
  DataPatch<bool> *visit_patch = new DataPatch<bool>(pixels, lines);
  bool **visit = visit_patch->get_data_lines_ptr();

  for(int line = 0; line < lines; line ++) {
    for(int pixel = 0; pixel < pixels; pixel ++) {
      visit[line][pixel] = false;
      region_map[line][pixel] = -1;
    }
  }
  
  queue<Point> workq;

  int region_id = 0;
  
  for(int ii = 0; ii < lines; ii ++) {
    for(int jj = 0; jj < pixels; jj ++) {
      if(input_regions[ii][jj] == -1) continue;
      if(visit[ii][jj]) continue;

      workq.push(Point(jj, ii));

      region_map[ii][jj] = region_id;
      visit[ii][jj] = true;

      int original_id = input_regions[ii][jj];

      while( !workq.empty() ) {
        Point point = workq.front();
        workq.pop();
        int line  = point.get_Y();
        int pixel = point.get_X();

        if(line > 0) {
	  if( !visit[line - 1][pixel] && input_regions[line - 1][pixel] == original_id) { 
	    workq.push(Point(pixel, line - 1));
	    visit[line - 1][pixel] = true;
	    region_map[line - 1][pixel] = region_id;
	  }
	}
        if(line < lines - 1) {
	  if( !visit[line + 1][pixel] && input_regions[line + 1][pixel] == original_id) { 
	    workq.push(Point(pixel, line + 1));
	    visit[line + 1][pixel] = true;
	    region_map[line + 1][pixel] = region_id;
	  }
	}
        if(pixel > 0 ) {
	  if( !visit[line][pixel - 1] && input_regions[line][pixel - 1] == original_id) { 
	    workq.push(Point(pixel - 1, line));
	    visit[line][pixel - 1] = true;
	    region_map[line][pixel - 1] = region_id;
	  }
	}
        if(pixel < pixels - 1 ) {
	  if( !visit[line][pixel + 1] && input_regions[line][pixel + 1] == original_id) { 
	    workq.push(Point(pixel + 1, line));
	    visit[line][pixel + 1] = true;
	    region_map[line][pixel + 1] = region_id;
	  }
	}
	
        if(line > 0 && pixel > 0) {
	  if( !visit[line - 1][pixel-1] && input_regions[line - 1][pixel-1] == original_id) { 
	    workq.push(Point(pixel-1, line - 1));
	    visit[line - 1][pixel-1] = true;
	    region_map[line - 1][pixel-1] = region_id;
	  }
	}
        if(line > 0 && pixel < pixels - 1) {
	  if( !visit[line - 1][pixel+1] && input_regions[line - 1][pixel+1] == original_id) { 
	    workq.push(Point(pixel+1, line - 1));
	    visit[line - 1][pixel+1] = true;
	    region_map[line - 1][pixel+1] = region_id;
	  }
	}
        if(line < lines - 1 && pixel > 0) {
	  if( !visit[line + 1][pixel-1] && input_regions[line + 1][pixel-1] == original_id) { 
	    workq.push(Point(pixel-1, line + 1));
	    visit[line + 1][pixel-1] = true;
	    region_map[line + 1][pixel-1] = region_id;
	  }
	}
        if(line < lines - 1 && pixel < pixels - 1) {
	  if( !visit[line + 1][pixel+1] && input_regions[line + 1][pixel+1] == original_id) { 
	    workq.push(Point(pixel+1, line + 1));
	    visit[line + 1][pixel+1] = true;
	    region_map[line + 1][pixel+1] = region_id;
	  }
	}

      }
      region_id ++;
    }
  }

  delete visit_patch;

  list<USPoint> *point_list = make_point_list(lines, pixels, region_map, region_id);
  for(int i = 0; i < lines; i++) {
    for(int j = 0; j < pixels; j++) {
      region_map[i][j] = -1;
    }
  }

  int nr_regions = region_id;

//  cerr << "nr_regions: " << nr_regions << endl;

  region_id = 0;
  for(int id = 0; id < nr_regions; id++) {

//cerr << "id: " << id << "  size: " << point_list[id].size() << "  start: " << *(point_list[id].begin()) << endl;

    if(point_list[id].size() < min_pixels_per_region) continue;
    for(list<USPoint>::iterator it = point_list[id].begin(); it != point_list[id].end(); it++) {
      USPoint point = *it;
      int line = point.y;
      int pixel = point.x;
      region_map[line][pixel] = region_id;
    }    
    region_id ++;
  }  
  
  for(int id = 0; id < nr_regions; id++) {
    point_list[id].clear();
  }
  delete[] point_list;
  
//  exit(0);

  return region_id;
}


