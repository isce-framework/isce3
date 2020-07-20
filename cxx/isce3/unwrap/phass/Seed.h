// ----------------------------------------------------------------------------
// Author:  Xiaoqing Wu
//

#pragma once

#include "Point.h"
#include "constants.h"

using namespace std;

struct Seed {
  int x;
  int y;
  int nr_2pi;
  int pc; // pixel count
};

void read_seed_file(char *seed_file, int & nr_seeds, Seed **seeds);
void write_seeds(char *seed_file, int nr_seeds, Seed *seeds, double cp = 0);
