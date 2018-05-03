//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2018

#ifndef ISCE_CORE_STATEVECTOR_H
#define ISCE_CORE_STATEVECTOR_H

#include <string>

// isce::core
#include "Constants.h"
#include "DateTime.h"

// Declaration
namespace isce {
    namespace core {
        class StateVector;
    }
}

// StateVector definition
class isce::core::StateVector {

    public:
        // Constructors
        StateVector() {}
      
        // Get state vector components 
        cartesian_t position() const { return _position; }
        cartesian_t velocity() const { return _velocity; }
        DateTime date() const { return _date; }
        // Set state vector components
        void position(cartesian_t & p) { _position = p; }
        void velocity(cartesian_t & v) { _velocity = v; }
        void date(const std::string & dstr) { _date = dstr; }

        // String formatted output for position and velocity
        inline std::string positionToString() const;
        inline std::string velocityToString() const;

        // Parse formatted string to save position and velocity
        inline void fromString(const std::string &, const std::string &);

    private:
        // Cartesian types for position and velocity
        cartesian_t _position;
        cartesian_t _velocity;
        DateTime _date;
};

// String formatted output for position
std::string isce::core::StateVector::
positionToString() const {
    std::stringstream output;
    output << _position[0] << " "
           << _position[1] << " "
           << _position[2];
    return output.str();
}

// String formatted output for velocity
std::string isce::core::StateVector::
velocityToString() const {
    std::stringstream output;
    output << _velocity[0] << " "
           << _velocity[1] << " "
           << _velocity[2];
    return output.str();
}

// Parse formatted string to save position/velocity components
void isce::core::StateVector::
fromString(const std::string & posString, const std::string & velString) {
    std::stringstream posStream(posString);
    std::stringstream velStream(velString);
    posStream >> _position[0] >> _position[1] >> _position[2];
    velStream >> _velocity[0] >> _velocity[1] >> _velocity[2];
}

#endif

// end of file
