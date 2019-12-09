// Copyright (c) 2017-, California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged.
// All rights reserved.
// 
// Author(s):
// 
//  ======================================================================
// 
//  FILENAME: sort.cc
// 
//  ======================================================================

#include "sort.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h> 

using namespace std;

void heapSort(int n, int *vals, int *indexes, int order)
{
  int a, tmp;
  //  for(a = 0; a < n; a ++)    indexes[a] = a;

  if(order == 0) {
    for (a = n/2; a > 0; a --) maxHeapify(vals, n, indexes, a);
    
    for (a = n-1; a > 0; a --) {
      tmp = indexes[0];
      indexes[0] = indexes[a];
      indexes[a] = tmp;
      maxHeapify(vals, a, indexes, 1);
    }
  }
  else {
    int *tmp_indexes = new int[n];
    for(a = 0; a < n; a++) tmp_indexes[a] = a;
    for (a = n/2; a > 0; a --) maxHeapify(vals, n, tmp_indexes, a);
    for (a = n-1; a > 0; a --) {
      tmp = tmp_indexes[0];
      tmp_indexes[0] = tmp_indexes[a];
      tmp_indexes[a] = tmp;
      maxHeapify(vals, a, tmp_indexes, 1);
    }
    
    // reverse .......
    for(a = 0; a < n; a++) indexes[a] = tmp_indexes[n - 1 - a];
    delete[] tmp_indexes;
  }
  return;
}

void heapSort(int n, double *vals, int *indexes, int order)
{
  int a, tmp;
  //  for(a = 0; a < n; a ++)    indexes[a] = a;

  if(order == 0) {
    for (a = n/2; a > 0; a --) maxHeapify(vals, n, indexes, a);
    
    for (a = n-1; a > 0; a --) {
      tmp = indexes[0];
      indexes[0] = indexes[a];
      indexes[a] = tmp;
      maxHeapify(vals, a, indexes, 1);
    }
  }
  else {
    int *tmp_indexes = new int[n];
    for(a = 0; a < n; a++) tmp_indexes[a] = a;
    for (a = n/2; a > 0; a --) maxHeapify(vals, n, tmp_indexes, a);
    for (a = n-1; a > 0; a --) {
      tmp = tmp_indexes[0];
      tmp_indexes[0] = tmp_indexes[a];
      tmp_indexes[a] = tmp;
      maxHeapify(vals, a, tmp_indexes, 1);
    }
    
    // reverse .......
    for(a = 0; a < n; a++) indexes[a] = tmp_indexes[n - 1 - a];
    delete[] tmp_indexes;
  }
  return;
}

void heapSort(int n, float *vals, int *indexes, int order)
{
  int a, tmp;
  //  for(a = 0; a < n; a ++)    indexes[a] = a;

  if(order == 0) {
    for (a = n/2; a > 0; a --) maxHeapify(vals, n, indexes, a);
    
    for (a = n-1; a > 0; a --) {
      tmp = indexes[0];
      indexes[0] = indexes[a];
      indexes[a] = tmp;
      maxHeapify(vals, a, indexes, 1);
    }
  }
  else {
    int *tmp_indexes = new int[n];
    for(a = 0; a < n; a++) tmp_indexes[a] = a;
    for (a = n/2; a > 0; a --) maxHeapify(vals, n, tmp_indexes, a);
    for (a = n-1; a > 0; a --) {
      tmp = tmp_indexes[0];
      tmp_indexes[0] = tmp_indexes[a];
      tmp_indexes[a] = tmp;
      maxHeapify(vals, a, tmp_indexes, 1);
    }
    
    // reverse .......
    for(a = 0; a < n; a++) indexes[a] = tmp_indexes[n - 1 - a];
    delete[] tmp_indexes;
  }
  return;
}

void maxHeapify(int *vals, int n, int *indexes, int root)
{ 
  int done = 0;
  int m = root * 2;
 
  int tmp;
  while ((m <= n) && ( !done )) {
    if ( (m == n) || ( vals[ indexes[m-1] ] > vals[ indexes[m] ] ) ) {
      ;
    } else {
      m++;
    }
 
    if ( vals[ indexes[root-1] ] < vals[ indexes[m-1] ] ) {
	tmp = indexes[root-1];
	indexes[root-1] = indexes[m-1];
	indexes[m-1] = tmp;
        root = m;
        m *= 2;
    } else 
    done = 1;
  }
  return;
}

void maxHeapify(double *vals, int n, int *indexes, int root)
{ 
  int done = 0;
  int m = root * 2;
 
  int tmp;
  while ((m <= n) && ( !done )) {
    if ( (m == n) || ( vals[ indexes[m-1] ] > vals[ indexes[m] ] ) ) {
      ;
    } else {
      m++;
    }
 
    if ( vals[ indexes[root-1] ] < vals[ indexes[m-1] ] ) {
	tmp = indexes[root-1];
	indexes[root-1] = indexes[m-1];
	indexes[m-1] = tmp;
        root = m;
        m *= 2;
    } else 
    done = 1;
  }
  return;
}

void maxHeapify(float *vals, int n, int *indexes, int root)
{ 
  int done = 0;
  int m = root * 2;
 
  int tmp;
  while ((m <= n) && ( !done )) {
    if ( (m == n) || ( vals[ indexes[m-1] ] > vals[ indexes[m] ] ) ) {
      ;
    } else {
      m++;
    }
 
    if ( vals[ indexes[root-1] ] < vals[ indexes[m-1] ] ) {
	tmp = indexes[root-1];
	indexes[root-1] = indexes[m-1];
	indexes[m-1] = tmp;
        root = m;
        m *= 2;
    } else 
    done = 1;
  }
  return;
}


void medium_filtering(int n, double *data, int win, double nodata)
{
  double *temp_data = new double[n];
  for(int i = 0; i < n; i++) temp_data[i] = data[i];
  double *work_data = new double[2*win + 1];
  int *indexes = new int[2*win + 1];
  for(int i = 0; i < n; i++) {
    int count = 0;
    for(int j = i - win; j <= i + win; j++) {
      if(j < 0 || j >= n) continue;
      if(temp_data[j] == nodata) continue;
      work_data[count] = temp_data[j];
      indexes[count] = count;
      count ++;
    }
    if(count == 0) continue;
    if(count == 1) {
      data[i] = work_data[0];
      continue;
    }
    heapSort(count, work_data, indexes);
    data[i] = work_data[indexes[count/2]];
  }
  delete[] work_data;
  delete[] temp_data;
  delete[] indexes;
}
void medium_filtering(int n, float *data, int win, float nodata)
{
  float *temp_data = new float[n];
  for(int i = 0; i < n; i++) temp_data[i] = data[i];
  double *work_data = new double[2*win + 1];
  int *indexes = new int[2*win + 1];
  for(int i = 0; i < n; i++) {
    int count = 0;
    for(int j = i - win; j <= i + win; j++) {
      if(j < 0 || j >= n) continue;
      if(temp_data[j] == nodata) continue;
      work_data[count] = temp_data[j];
      indexes[count] = count;
      count ++;
    }
    if(count == 0) continue;
    if(count == 1) {
      data[i] = work_data[0];
      continue;
    }
    heapSort(count, work_data, indexes);
    data[i] = work_data[indexes[count/2]];
  }
  delete[] work_data;
  delete[] temp_data;
  delete[] indexes;
}


float median_of_medians(int n, float *data, int rank)
{
  int len = 5;
  int *indexes = new int[len];
  float *vals = new float[len];
  int nr_sublists = (n - 1)/len + 1;
  float *medians = new float[nr_sublists];

  float pivot;

  for(int s = 0; s < nr_sublists; s++) {  
    int count = 0;
    for(int i = 0; i < len; i++) {
      indexes[i] = i;   
      if(s*len + i < n) {
	vals[i] = data[s*len + i];
//        cerr << vals[i]	 << " " << endl;
	count ++;
      }
      else break;
    }
    heapSort(count, vals, indexes);
    medians[s] = vals[indexes[count/2]];
//    cerr << "\ns: " << s << "  count: " << count<< "  median: " << medians[s] << endl;
  }
//  cerr << endl;

  if(nr_sublists <= len) {  
    //int count = 0;
    for(int i = 0; i < nr_sublists; i++) {
      indexes[i] = i;  
      vals[i] = medians[i];
    }
    heapSort(nr_sublists, vals, indexes);
    pivot = vals[indexes[nr_sublists/2]];

//    cerr << "pivot: " << pivot << endl;

  }
  else {
    pivot = median_of_medians(nr_sublists, medians, nr_sublists/2);
  }
    
  int low_count = 0;
  int high_count = 0;
  float *low = new float[n];
  float *high = new float[n];
  for(int i = 0; i < n; i++) {
    if(data[i] < pivot) {
      low[low_count++] = data[i];
    }
    if(data[i] > pivot) {
      high[high_count++] = data[i];
    }
  }

  if(rank < low_count) return median_of_medians(low_count, low, rank);
  else if(rank > low_count) return median_of_medians(high_count, high, rank - low_count - 1);
  else return pivot;
}   


float median_use_histogram(int n, float *data, int start_count, int total_size)
{
  if(start_count == 0 && total_size == 0) total_size = n;
 // if(n <= 2) exit(0);
  int print_flag = 0;
  if(print_flag) cerr << "n: " << n << "  start_count: " << start_count << "  total_size: " << total_size << endl;

  if(n < 10) {

//    double aver = 0;
//    for(int i = 0; i < n; i++) {
//      aver += data[i];
//    }
//    return (float)(aver/(double)n);    

    int *indexes = new int[n];
    for(int i = 0; i < n; i++) {
      indexes[i] = i;
      if(print_flag) cerr << "i: " << i << "  data: " << data[i] << endl;
    }
    if(n <= 2) {
      if(n == 2) {
	if(data[1] < data[0]) {
	  indexes[0] = 1;
	  indexes[1] = 0;
	}
      }
    }
    else {
      heapSort(n, data, indexes);
    }
    int tot = start_count;
    float ret = 0;
    for(int i = 0; i < n; i++) {
      tot ++;
if(print_flag) cerr << "i: " << i << "  tot: " << tot << "  data: " << data[indexes[i]] << endl;
      if(tot > total_size/2) {
	ret = data[indexes[i]];
	break;
      }
    }
    delete[] indexes;
    return ret;
  }

  float minv = 1.0e30;
  float maxv = -minv;

  for(int i = 0; i < n; i++) {
    float x = data[i];
    if(x < minv) minv = x;
    if(x > maxv) maxv = x;
    if(print_flag && n < 100) cerr << "i: " << i << "  data: " << data[i] << endl;
   
  }
  int levels = sqrt(n);
  if(levels < 10) levels = 10;
  double step = (maxv - minv)/(double)levels;
  if(step < 0.000001) return (float)((maxv + minv)/2.0);
  step += step * 0.01;

if(print_flag)  cerr << "minv: " << minv << "  maxv: " << maxv << "  step: " << step << "  levels: " << levels << endl;

  int *histogram = new int[levels];
  for(int i = 0; i < levels; i++) {
    histogram[i] = 0;
  }
 

  for(int i = 0; i < n; i++) {
    float x = data[i];
    int level = (x - minv)/step;
    histogram[level] ++;
if(print_flag && level == 1 && n == 19) cerr << "level: " << level << "  x: " << x << "  histo: " << histogram[level] << endl;
  }  

  for(int level = 0; level < levels; level++) {
    if(print_flag) cerr << "level: " << level << "  histo: " << histogram[level] << endl;
  }
  
  
  int tot = start_count;
  int level0 = 0;
  for(int level = 0; level < levels; level++) {
    tot += histogram[level];
    if(print_flag) cerr << "level: " << level << "  tot: " << tot << endl;
    if(tot > total_size/2) {
      level0 = level;
      break;
    }
  }

  if(print_flag) cerr << "level0: " << level0 << endl;

  int new_n = histogram[level0];
  float *new_data = new float[new_n];
  int count = 0;
  for(int i = 0; i < n; i ++) {
    float x = data[i];
    int level = (x - minv)/step;
    if(level == level0) {
      new_data[count] = x;
    if(print_flag && (new_n == 19 || new_n == 3)) cerr << "i: " << i << "  count: " << count << "  new_data: " << new_data[count] << "  new_n: " << new_n << endl;
      count ++;
      if(count == new_n) break;
    }

  }

  if(level0 > 0) start_count = tot - histogram[level0];

  delete[] histogram;
  float ret = median_use_histogram(new_n, new_data, start_count, total_size);
  delete[] new_data;
  return ret;
}
