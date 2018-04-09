//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef ISCE_CORE_PEG_H
#define ISCE_CORE_PEG_H

// Declaration
namespace isce {
    namespace core {
        struct Peg;
    }
}

// Peg declaration
struct isce::core::Peg {
    double lat;
    double lon;
    double hdg;

    Peg(double lt, double ln, double hd) : lat(lt), lon(ln), hdg(hd) {}
    Peg() : Peg(0.,0.,0.) {}
    Peg(const Peg &p) : lat(p.lat), lon(p.lon), hdg(p.hdg) {}
    inline Peg& operator=(const Peg&);
};

isce::core::Peg & isce::core::Peg::
operator=(const Peg &rhs) {
    lat = rhs.lat;
    lon = rhs.lon;
    hdg = rhs.hdg;
    return *this;
}

#endif
