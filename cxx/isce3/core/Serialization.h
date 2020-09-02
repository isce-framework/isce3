//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2018

/** \file core/Serialization.h
 *
 * Serialization functions for isce3::core objects. */

#pragma once

#include <iostream>
#include <memory>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/archives/xml.hpp>
#include <sstream>
#include <stdexcept>
#include <vector>

// pyre
#include <pyre/journal.h>

// isce3::core
#include <isce3/core/Attitude.h>
#include <isce3/core/DateTime.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/Metadata.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/Poly2d.h>
#include <isce3/core/LUT1d.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/StateVector.h>
#include <isce3/core/TimeDelta.h>

// isce3::io
#include <isce3/io/IH5.h>
#include <isce3/io/Serialization.h>

//! The isce namespace
namespace isce3 {
//! The isce3::core namespace
namespace core {

// Archiving any isce3::core object by pointer
template <typename T>
inline void load_archive(std::string metadata, char * objectTag, T * object)
{
    std::stringstream metastream;
    metastream << metadata;
    cereal::XMLInputArchive archive(metastream);
    archive(cereal::make_nvp(objectTag, (*object)));
}

// Archiving any isce3::core object by reference
template <typename T>
inline void load_archive_reference(std::string metadata, char * objectTag, T & object)
{
    std::stringstream metastream;
    metastream << metadata;
    cereal::XMLInputArchive archive(metastream);
    archive(cereal::make_nvp(objectTag, object));
}

// ------------------------------------------------------------------------
// Serialization for Ellipsoid
// ------------------------------------------------------------------------

template<class Archive>
inline void save(Archive & archive, const Ellipsoid & ellps) {
    archive(cereal::make_nvp("a", ellps.a()), cereal::make_nvp("e2", ellps.e2()));
}

template<class Archive>
inline void load(Archive & archive, Ellipsoid & ellps)
{
    double a, e2;
    archive(cereal::make_nvp("a", a), cereal::make_nvp("e2", e2));
    ellps.a(a);
    ellps.e2(e2);
}

/**
 * Load Ellipsoid parameters from HDF5.
 *
 * @param[in] group         HDF5 group object.
 * @param[in] ellps         Ellipsoid object to be configured.
 */
inline void loadFromH5(isce3::io::IGroup & group, Ellipsoid & ellps)
{
    // Read data
    std::vector<double> ellpsData;
    isce3::io::loadFromH5(group, "ellipsoid", ellpsData);
    // Set ellipsoid properties
    ellps.a(ellpsData[0]);
    ellps.e2(ellpsData[1]);
}

// ------------------------------------------------------------------------
// Serialization for Orbit
// ------------------------------------------------------------------------

template <class Archive>
inline void save(Archive & archive, const Orbit & orbit) {
    archive(cereal::make_nvp("StateVectors", orbit.getStateVectors()),
            cereal::make_nvp("InterpMethod", orbit.interpMethod()));
}

template <class Archive>
inline void load(Archive & archive, Orbit & orbit)
{
    std::vector<StateVector> statevecs;
    OrbitInterpMethod interp_method;
    archive(cereal::make_nvp("StateVectors", statevecs),
            cereal::make_nvp("InterpMethod", interp_method));
    orbit.referenceEpoch(statevecs.at(0).datetime);
    orbit.setStateVectors(statevecs);
    orbit.interpMethod(interp_method);
}

/**
 * \brief Load orbit data from HDF5 product.
 *
 * @param[in] group         HDF5 group object.
 * @param[in] orbit         Orbit object to be configured.
 */
inline void loadFromH5(isce3::io::IGroup & group, Orbit & orbit)
{
    // open datasets
    isce3::io::IDataSet time_ds = group.openDataSet("time");
    isce3::io::IDataSet pos_ds = group.openDataSet("position");
    isce3::io::IDataSet vel_ds = group.openDataSet("velocity");

    // get dataset dimensions
    std::vector<int> time_dims = time_ds.getDimensions();
    std::vector<int> pos_dims = pos_ds.getDimensions();
    std::vector<int> vel_dims = vel_ds.getDimensions();

    // check validity
    if (time_dims.size() != 1 || pos_dims.size() != 2 || vel_dims.size() != 2) {
        throw std::runtime_error("unexpected orbit state vector dims");
    }
    if (pos_dims[1] != 3 || vel_dims[1] != 3) {
        throw std::runtime_error("unexpected orbit position/velocity vector size");
    }
    std::size_t size = time_dims[0];
    if (pos_dims[0] != size || vel_dims[0] != size) {
        throw std::runtime_error("mismatched orbit state vector component sizes");
    }

    // read orbit data
    std::vector<double> time(size);
    std::vector<double> pos(3 * size);
    std::vector<double> vel(3 * size);
    time_ds.read(time);
    pos_ds.read(pos);
    vel_ds.read(vel);

    // get reference epoch
    DateTime reference_epoch = isce3::io::getRefEpoch(group, "time");

    // convert to state vectors
    std::vector<StateVector> statevecs(size);
    for (std::size_t i = 0; i < size; ++i) {
        statevecs[i].datetime = reference_epoch + TimeDelta(time[i]);
        statevecs[i].position = { pos[3*i+0], pos[3*i+1], pos[3*i+2] };
        statevecs[i].velocity = { vel[3*i+0], vel[3*i+1], vel[3*i+2] };
    }

    // build orbit
    orbit.referenceEpoch(reference_epoch);
    orbit.setStateVectors(statevecs);

    // get interp method
    std::string interp_method = "Hermite";
    if (isce3::io::exists(group, "interpMethod")) {
        isce3::io::loadFromH5(group, "interpMethod", interp_method);
    }

    if (interp_method == "Hermite") {
        orbit.interpMethod(OrbitInterpMethod::Hermite);
    }
    else if (interp_method == "Legendre") {
        orbit.interpMethod(OrbitInterpMethod::Legendre);
    }
    else {
        throw std::runtime_error("unexpected orbit interpolation method '" + interp_method + "'");
    }
}

/**
 * \brief Save orbit data to HDF5 product.
 *
 * @param[in] group         HDF5 group object.
 * @param[in] orbit         Orbit object to be saved.
 */
inline void saveToH5(isce3::io::IGroup & group, const Orbit & orbit)
{
    // convert times to vector, get flattened position, velocity
    std::vector<double> time(orbit.size());
    std::vector<double> pos(3 * orbit.size());
    std::vector<double> vel(3 * orbit.size());

    for (int i = 0; i < orbit.size(); ++i) {
        time[i] = orbit.time(i);

        pos[3*i+0] = orbit.position(i)[0];
        pos[3*i+1] = orbit.position(i)[1];
        pos[3*i+2] = orbit.position(i)[2];

        vel[3*i+0] = orbit.velocity(i)[0];
        vel[3*i+1] = orbit.velocity(i)[1];
        vel[3*i+2] = orbit.velocity(i)[2];
    }

    // position/velocity data dims
    std::size_t size = orbit.size();
    std::array<std::size_t, 2> dims = {size, 3};

    // interp method
    std::string interp_method;
    if (orbit.interpMethod() == OrbitInterpMethod::Hermite) {
        interp_method = "Hermite";
    }
    else if (orbit.interpMethod() == OrbitInterpMethod::Legendre) {
        interp_method = "Legendre";
    }
    else {
        throw std::runtime_error("unexpected orbit interpolation method");
    }

    // serialize
    isce3::io::saveToH5(group, "time", time);
    isce3::io::setRefEpoch(group, "time", orbit.referenceEpoch());
    isce3::io::saveToH5(group, "position", pos, dims, "meters");
    isce3::io::saveToH5(group, "velocity", vel, dims, "meters per second");
    isce3::io::saveToH5(group, "interpMethod", interp_method);
}

/**
 * \brief Save Euler angle data to HDF5 product.
 *
 * @param[in] group         HDF5 group object.
 * @param[in] euler         EulerAngles object to be save.
inline void saveToH5(isce3::io::IGroup & group, const EulerAngles & euler)
{
    // Create vector to store all data (convert angles to degrees)
    const double deg = 180.0 / M_PI;
    std::vector<double> angles(euler.nVectors() * 3);
    for (size_t i = 0; i < euler.nVectors(); ++i) {
        angles[i*3 + 0] = deg * euler.yaw()[i];
        angles[i*3 + 1] = deg * euler.pitch()[i];
        angles[i*3 + 2] = deg * euler.roll()[i];
    }

    // Save angles
    std::array<size_t, 2> dims = {euler.nVectors(), 3};
    isce3::io::saveToH5(group, "eulerAngles", angles, dims, "degrees");

    // Save time and reference epoch attribute
    isce3::io::saveToH5(group, "time", euler.time());
    isce3::io::setRefEpoch(group, "time", euler.refEpoch());
}
 */

// ------------------------------------------------------------------------
// Serialization for Attitude
// ------------------------------------------------------------------------

inline void loadFromH5(isce3::io::IGroup& group, Attitude& att)
{
    // load data from file
    DateTime epoch = isce3::io::getRefEpoch(group, "time");
    std::vector<double> time, packed_quat;
    isce3::io::loadFromH5(group, "time", time);
    isce3::io::loadFromH5(group, "quaternions", packed_quat);
    // convert quaternion representation
    int n = packed_quat.size() / 4;
    std::vector<isce3::core::Quaternion> quat(n);
    for (int i = 0; i < n; ++i) {
        const double* q = &packed_quat[i * 4];
        // XXX Careful to use 4-arg ctor (not array) to avoid mixing up order of
        // elements.  Want real part followed by bivector part.
        quat[i] = isce3::core::Quaternion(q[0], q[1], q[2], q[3]);
    }
    // use ctor for remaining checks
    att = isce3::core::Attitude(time, quat, epoch);
}

inline void saveToH5(isce3::io::IGroup & group, const Attitude& att)
{
    // Flatten quaternion vector.
    int n = att.size();
    std::vector<double> qflat(n * 4);
    auto qvec = att.quaternions();
    for (int i = 0; i < n; ++i) {
        double *q = &qflat[i * 4];
        // XXX Don't use internal Quaternion array since it uses a
        // different storage order!
        q[0] = qvec[i].w();
        q[1] = qvec[i].x();
        q[2] = qvec[i].y();
        q[3] = qvec[i].z();
    }
    // Save to disk.
    isce3::io::saveToH5(group, "time", att.time());
    isce3::io::setRefEpoch(group, "time", att.referenceEpoch());
    std::array<size_t, 2> dims = {static_cast<size_t>(n), 4};
    isce3::io::saveToH5(group, "quaternions", qflat, dims);
    // TODO convert and save EulerAngles
}

// ------------------------------------------------------------------------
// Serialization for Metadata
// ------------------------------------------------------------------------

template <class Archive>
inline void save(Archive & archive, const Metadata & meta)
{
    archive(cereal::make_nvp("width", meta.width),
            cereal::make_nvp("length", meta.length),
            cereal::make_nvp("numberRangeLooks", meta.numberRangeLooks),
            cereal::make_nvp("numberAzimuthLooks", meta.numberAzimuthLooks),
            cereal::make_nvp("slantRangePixelSpacing", meta.slantRangePixelSpacing),
            cereal::make_nvp("rangeFirstSample", meta.rangeFirstSample),
            cereal::make_nvp("lookSide", meta.lookSide),
            cereal::make_nvp("prf", meta.prf),
            cereal::make_nvp("radarWavelength", meta.radarWavelength),
            cereal::make_nvp("pegHeading", meta.pegHeading),
            cereal::make_nvp("pegLatitude", meta.pegLatitude),
            cereal::make_nvp("pegLongitude", meta.pegLongitude),
            cereal::make_nvp("chirpSlope", meta.chirpSlope),
            cereal::make_nvp("antennaLength", meta.antennaLength),
            cereal::make_nvp("sensingStart", meta.sensingStart.isoformat()));
}

template <class Archive>
inline void load(Archive & archive, Metadata & meta)
{
    std::string sensingStart;
    archive(cereal::make_nvp("width", meta.width),
            cereal::make_nvp("length", meta.length),
            cereal::make_nvp("numberRangeLooks", meta.numberRangeLooks),
            cereal::make_nvp("numberAzimuthLooks", meta.numberAzimuthLooks),
            cereal::make_nvp("slantRangePixelSpacing", meta.slantRangePixelSpacing),
            cereal::make_nvp("rangeFirstSample", meta.rangeFirstSample),
            cereal::make_nvp("lookSide", meta.lookSide),
            cereal::make_nvp("prf", meta.prf),
            cereal::make_nvp("radarWavelength", meta.radarWavelength),
            cereal::make_nvp("pegHeading", meta.pegHeading),
            cereal::make_nvp("pegLatitude", meta.pegLatitude),
            cereal::make_nvp("pegLongitude", meta.pegLongitude),
            cereal::make_nvp("chirpSlope", meta.chirpSlope),
            cereal::make_nvp("antennaLength", meta.antennaLength),
            cereal::make_nvp("sensingStart", sensingStart));
    meta.sensingStart = sensingStart;
}

// ------------------------------------------------------------------------
// Serialization for Poly2d
// ------------------------------------------------------------------------

// Definition for Poly2d
template <class Archive>
inline void serialize(Archive & archive, Poly2d & poly)
{
    archive(cereal::make_nvp("rangeOrder", poly.rangeOrder),
            cereal::make_nvp("azimuthOrder", poly.azimuthOrder),
            cereal::make_nvp("rangeMean", poly.rangeMean),
            cereal::make_nvp("azimuthMean", poly.azimuthMean),
            cereal::make_nvp("rangeNorm", poly.rangeNorm),
            cereal::make_nvp("azimuthNorm", poly.azimuthNorm),
            cereal::make_nvp("coeffs", poly.coeffs));
}

/**
 * \brief Load polynomial coefficients from HDF5 product.
 *
 * @param[in] group         HDF5 group object.
 * @param[in] poly          Poly2d to be configured.
 * @param[in] name          Dataset name within group.
 */
inline void loadFromH5(isce3::io::IGroup & group, Poly2d & poly, std::string name)
{
    // Configure the polynomial coefficients
    isce3::io::loadFromH5(group, name, poly.coeffs);

    // Set other polynomial properties
    poly.rangeOrder = poly.coeffs.size() - 1;
    poly.azimuthOrder = 0;
    poly.rangeMean = 0.0;
    poly.azimuthMean = 0.0;
    poly.rangeNorm = 1.0;
    poly.azimuthNorm = 1.0;
}

// ------------------------------------------------------------------------
// Serialization for LUT2d (specifically for calibration grids)
// ------------------------------------------------------------------------

/**
 * \brief Load LUT2d data from HDF5 product.
 *
 * @param[in] group         HDF5 group object.
 * @param[in] dsetName      Dataset name within group
 * @param[in] lut           LUT2d to be configured.
 */
template <typename T>
inline void loadCalGrid(isce3::io::IGroup & group, const std::string & dsetName,
                        isce3::core::LUT2d<T> & lut)
{
    // Load coordinates
    std::valarray<double> slantRange, zeroDopplerTime;
    isce3::io::loadFromH5(group, "slantRange", slantRange);
    isce3::io::loadFromH5(group, "zeroDopplerTime", zeroDopplerTime);

    // Load LUT2d data in matrix
    isce3::core::Matrix<T> matrix(zeroDopplerTime.size(), slantRange.size());
    isce3::io::loadFromH5(group, dsetName, matrix);

    // Set in lut
    lut.setFromData(slantRange, zeroDopplerTime, matrix);
}

/**
 * \brief Save LUT2d data to HDF5 product.
 *
 * @param[in] group         HDF5 group object.
 * @param[in] dsetName      Dataset name within group
 * @param[in] lut           LUT2d to be saved.
 * @param[in] units         Units of LUT2d data.
 */
template <typename T>
inline void saveCalGrid(isce3::io::IGroup & group,
                        const std::string & dsetName,
                        const isce3::core::LUT2d<T> & lut,
                        const isce3::core::DateTime & refEpoch,
                        const std::string & units = "")
{
    // Generate uniformly spaced X (slant range) coordinates
    const double x0 = lut.xStart();
    const double x1 = x0 + lut.xSpacing() * (lut.width() - 1.0);
    const std::vector<double> x = isce3::core::linspace(x0, x1, lut.width());

    // Generate uniformly spaced Y (zero Doppler time) coordinates
    const double y0 = lut.yStart();
    const double y1 = y0 + lut.ySpacing() * (lut.length() - 1.0);
    const std::vector<double> y = isce3::core::linspace(y0, y1, lut.length());

    // Save coordinates
    isce3::io::saveToH5(group, "slantRange", x, "meters");
    isce3::io::saveToH5(group, "zeroDopplerTime", y);
    isce3::io::setRefEpoch(group, "zeroDopplerTime", refEpoch);

    // Save LUT2d data
    isce3::io::saveToH5(group, dsetName, lut.data(), units);
}

// ------------------------------------------------------------------------
// Serialization for LUT1d
// ------------------------------------------------------------------------

// Serialization save method
template <class Archive, typename T>
inline void save(Archive & archive, LUT1d<T> const & lut)
{
    // Copy LUT data from valarrays to vectors
    std::vector<double> coords(lut.size());
    std::vector<T> values(lut.size());
    auto v_coords = lut.coords();
    auto v_values = lut.values();
    coords.assign(std::begin(v_coords), std::end(v_coords));
    values.assign(std::begin(v_values), std::end(v_values));
    // Archive
    archive(cereal::make_nvp("Coords", coords),
            cereal::make_nvp("Values", values));
}

// Serialization load method
template<class Archive, typename T>
inline void load(Archive & archive, LUT1d<T> & lut)
{
    // Create vector for loading results
    std::vector<double> coords;
    std::vector<T> values;
    // Load the archive
    archive(cereal::make_nvp("Coords", coords),
            cereal::make_nvp("Values", values));
    // Copy vector to LUT valarrays
    std::valarray<double> v_coords(coords.data(), coords.size());
    std::valarray<T> v_values(values.data(), values.size());
    lut.coords(v_coords);
    lut.values(v_values);
}

/**
 * \brief Load polynomial coefficients from HDF5 product.
 *
 * @param[in] group         HDF5 group object.
 * @param[in] poly          Poly2d to be configured.
 * @param[in] name          Dataset name within group.
 */
template <typename T>
inline void loadFromH5(isce3::io::IGroup & group, LUT1d<T> & lut,
                       std::string name_coords, std::string name_values)
{
    // Valarrays for storing results
    std::valarray<double> x, y;
    // Load the LUT values
    isce3::io::loadFromH5(group, name_values, y);
    // Load the LUT coordinates
    isce3::io::loadFromH5(group, name_coords, x);
    // Set LUT data
    lut.coords(x);
    lut.values(y);
}

// ------------------------------------------------------------------------
// Serialization for StateVector
// ------------------------------------------------------------------------

// Serialization save method
template<class Archive>
inline void save(Archive & archive, const StateVector & sv)
{
    // Serialize position vector to string as whitespace-delimited values
    std::stringstream pos_stream;
    pos_stream << sv.position[0] << " " << sv.position[1] << " " << sv.position[2];
    std::string position_string = pos_stream.str();
    // Serialize velocity vector to string as whitespace-delimited values
    std::stringstream vel_stream;
    vel_stream << sv.velocity[0] << " " << sv.velocity[1] << " " << sv.velocity[2];
    std::string velocity_string = vel_stream.str();
    // Archive
    archive(cereal::make_nvp("Time", sv.datetime.isoformat()),
            cereal::make_nvp("Position", position_string),
            cereal::make_nvp("Velocity", velocity_string));
}

// Serialization load method
template<class Archive>
inline void load(Archive & archive, StateVector & sv)
{
    // Make strings for position, velocity, and datetime
    std::string position_string, velocity_string, datetime_string;
    // Load the archive
    archive(cereal::make_nvp("Time", datetime_string),
            cereal::make_nvp("Position", position_string),
            cereal::make_nvp("Velocity", velocity_string));
    // Send datetime string to datetime object parser
    sv.datetime = datetime_string;
    // De-serialize position vector from stringstream
    std::stringstream pos_stream (position_string);
    pos_stream >> sv.position[0] >> sv.position[1] >> sv.position[2];
    // De-serialize velocity vector from stringstream
    std::stringstream vel_stream (velocity_string);
    vel_stream >> sv.velocity[0] >> sv.velocity[1] >> sv.velocity[2];
}

}
}
