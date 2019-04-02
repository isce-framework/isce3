// ----------------------------------------------------------------------------
// Author:  Xiaoqing Wu
//


#include "Seed.h"
#include "sort.h"
#include "constants.h"
                        
#include <stdio.h>                        
#include <string.h>       


void read_seed_file(char *seed_file, int & nr_seeds, Seed **seeds) 
{
  char *st = NULL, mystring[1024];
  size_t size;
  nr_seeds = 0;
  FILE *fp = fopen(seed_file, "r");
  rewind(fp);
  while(1) {
    st = fgets(mystring, 1023, fp);
    if(st) nr_seeds ++;
    else break;
  }
  rewind(fp);

  nr_seeds --;
  
  int x, y, n;

  Seed *ret_seeds = new Seed[nr_seeds];

  st = fgets(mystring, 1023, fp);
  for(int id = 0; id < nr_seeds; id++) {
    st = fgets(mystring, 1023, fp);
    sscanf(st, "%d %d %d\n", &x, &y, &n);
    ret_seeds[id].x = x;
    ret_seeds[id].y = y;
    ret_seeds[id].nr_2pi = n;
  }

  *seeds = ret_seeds;
  fclose(fp);
}

void write_seeds(char *seed_file, int nr_seeds, Seed *seeds, double cp)
{
    int *indexes = new int[nr_seeds];
    double *y_locations = new double[nr_seeds];
    for(int i = 0; i < nr_seeds; i++) {
      indexes[i] = i;
      y_locations[i] = seeds[i].y;
    }
    heapSort (nr_seeds, y_locations, indexes, 0);

    FILE *fp = fopen(seed_file, "w");
    fprintf(fp, "     x       y   2pi  with const_phase of %lf \n", cp);
    for(int i = 0; i < nr_seeds; i++) {
      int index = indexes[i];
      fprintf(fp, "%7d %7d %3d\n", seeds[index].x, seeds[index].y, seeds[index].nr_2pi); 
    }
    delete[] indexes;
    delete[] y_locations;
    fclose(fp);
}

