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

/** Building block data structure for serializing Orbit
 *
 * This datastructure is used at the interface of Orbit to XML / HDF5. 
 * For computation, the information is copied into a linearized vector
 * of time, position and velocity inside the Orbit Object*/
class isce::core::StateVector {

    public:
        /** Empty constructor*/
        StateVector() {}
      
        /** Return position in m*/
        cartesian_t position() const { return _position; }

        /** Return velocity in m/s*/
        cartesian_t velocity() const { return _velocity; }

        /** Return DateTime */
        DateTime date() const { return _date; }

        /** Set position
         *
         * @param[in] pos ECEF position in m*/
        void position(cartesian_t & p) { _position = p; }

        /** Set velocity
         *
         * @param[in] vel ECEF velocity in m/s*/
        void velocity(cartesian_t & v) { _velocity = v; }

        /** Set time tag
         *
         * @param[in] dstr Datetime string in ISO-8601*/
        void date(const std::string & dstr) { _date = dstr; }

        /** Serialize position to string*/
        inline std::string positionToString() const;

        /** Serialize velocity to string*/
        inline std::string velocityToString() const;

        /** Parse formatted string to save position and velocity*/
        inline void fromString(const std::string &, const std::string &);

    private:
        /** ECEF position in m*/
        cartesian_t _position;

        /** ECEF velocity in m/s*/
        cartesian_t _velocity;

        /** Time tag */
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
