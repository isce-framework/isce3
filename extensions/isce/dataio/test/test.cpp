#include <iostream>
#include <cmath>
#include <vector>
#include "../Raster.h"

int main() {

  isce::dataio::Raster img = isce::dataio::Raster("kz.bin", false);

  int bw = 10;
  std::vector<float> data(img.getWidth() * bw);
  
  printf("%d %d \n", img.getLength(), img.getWidth());

  img.getBlock(data, 1, 1, bw, img.getWidth());

  for (auto i: data) std::cout << i << ' ';
  
  return 0;
}
