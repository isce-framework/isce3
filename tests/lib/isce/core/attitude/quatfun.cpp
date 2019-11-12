//
// Author: Brian Hawkins
// Copyright 2019
//

#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include <gtest/gtest.h>

#include "isce/core/Orbit.h"
#include "isce/core/Quaternion.h"
#include "isce/core/Utilities.h"

using isce::core::Quaternion;
using isce::core::cartesian_t;

Quaternion
// load_quat(const std::string filename)
load_quat(const char *filename)
{
    // count lines
    FILE *fp = fopen(filename, "r");
    long n = 0;
    size_t nbuf = 0;
    char *line = NULL;
    while ((getline(&line, &nbuf, fp) >= 0)) {
        ++n;
    }
    fseek(fp, 0, 0);
    
    // parse data
    std::vector<double> t(n), q(4*n);
    for (long i=0; i<n; ++i) {
        long k = 4*i;
        fscanf(fp, "%lg%lg%lg%lg%lg\n", &t[i], &q[k], &q[k+1], &q[k+2], &q[k+3]);
    }
    fclose(fp);
    
    // build object and return
    Quaternion out(t, q);
    return out;
}

int main(int argc, char * argv[]) {
    auto q = load_quat("/mnt/ssd/work/quaternion.txt");
    double t=101.0, yaw=0, pitch=0, roll=0;
    q.ypr(t, yaw, pitch, roll);
    printf("%g %g %g %g\n", t, yaw, pitch, roll);
    
    isce::core::Orbit orb;
    orb.loadFromHDR("/mnt/ssd/work/orbit_dt.txt");
    
    cartesian_t pos, vel;
    int i = orb.nVectors / 2;
    printf("i = %d\n", i);
    orb.getStateVector(i, t, pos, vel);
    // std::cout << t << " " << pos[0] << " " << vel[0] << std::endl;
    printf("%g %g %g\n", t, pos[0], vel[0]);
    // orb.printOrbit();
    return 0;
}
