#pragma once

#include <iostream>
#include <memory>

#include "Constants.h"
#include "Ellipsoid.h"
#include "Vector.h"

#include <isce3/except/Error.h>

namespace isce3 { namespace core {

/**
 * Abstract base class for individual projections
 *
 * Internally, every derived class is expected to provide two functions.
 * forward - To convert llh (radians) to expected projection system
 * inverse - To convert expected projection system to llh (radians)
 */
class ProjectionBase {
    /** @internal Ellipsoid object for projections - currently only WGS84 */
    Ellipsoid _ellipse;

    /**
     * @internal
     * Type of projection system. This can be used to check if projection
     * systems are equal Private member and should not be modified after
     * initialization
     */
    int _epsgcode;

public:
    /**
     * Value constructor with EPSG code as input. Ellipsoid is always
     * initialized to standard WGS84 ellipse.
     */
    ProjectionBase(int code)
        : _ellipse(), _epsgcode(code)
    {}

    /** Return EPSG code */
    int code() const { return _epsgcode; }

    /** Return underlying ellipsoid */
    const Ellipsoid& ellipsoid() const { return _ellipse; }

    /** Print function for debugging */
    virtual void print() const = 0;

    /**
     * Function for transforming from LLH. This is similar to fwd or
     * fwd3d in PROJ.4
     *
     * @param[in] llh Lon/Lat/Height - Lon and Lat are in radians
     * @param[out] xyz Coordinates in specified projection system
     */
    virtual int forward(const Vec3& llh, Vec3& xyz) const = 0;

    /**
     * Function for transforming from LLH. This is similar to fwd or
     * fwd3d in PROJ.4
     *
     * @param[in] llh Lon/Lat/Height - Lon and Lat are in radians
     * @returns Coordinates in specified projection system
     */
    Vec3 forward(const Vec3& llh) const
    {
        Vec3 xyz;
        const int ret = forward(llh, xyz);
        if (ret != 0) {
            throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                    "Forward projection transformation failed");
        }
        return xyz;
    }

    /**
     * Function for transforming to LLH. This is similar to inv or inv3d in
     * PROJ.4
     *
     * @param[in] xyz Coordinates in specified projection system
     * @param[out] llh Lat/Lon/Height - Lon and Lat are in radians
     */
    virtual int inverse(const Vec3& xyz, Vec3& llh) const = 0;

    /**
     * Function for transforming to LLH. This is similar to inv or inv3d in
     * PROJ.4
     *
     * @param[in] xyz Coordinates in specified projection system
     * @returns Lat/Lon/Height - Lon and Lat are in radians
     */
    Vec3 inverse(const Vec3& xyz) const
    {
        Vec3 llh;
        const int ret = inverse(xyz, llh);
        if (ret != 0) {
            throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                    "Inverse projection transformation failed");
        }
        return llh;
    }

    virtual ~ProjectionBase() = default;
};

/** Standard WGS84 Lon/Lat Projection extension of ProjBase - EPSG:4326 */
class LonLat : public ProjectionBase {
public:
    LonLat() : ProjectionBase(4326) {}

    /** @copydoc ProjectionBase::print() */
    void print() const override;

    int forward(const Vec3&, Vec3&) const override;

    int inverse(const Vec3&, Vec3&) const override;
};

inline void LonLat::print() const
{
    std::cout << "Projection: LatLon" << std::endl
              << "EPSG: " << code() << std::endl;
}

inline int LonLat::forward(const Vec3& in, Vec3& out) const
{
    out[0] = in[0] * 180.0 / M_PI;
    out[1] = in[1] * 180.0 / M_PI;
    out[2] = in[2];
    return 0;
}

inline int LonLat::inverse(const Vec3& in, Vec3& out) const
{
    out[0] = in[0] * M_PI / 180.0;
    out[1] = in[1] * M_PI / 180.0;
    out[2] = in[2];
    return 0;
}

/** Standard WGS84 ECEF coordinates extension of ProjBase - EPSG:4978 */
class Geocent : public ProjectionBase {
public:
    Geocent() : ProjectionBase(4978) {}

    /** @copydoc ProjectionBase::print() */
    void print() const;

    /** This is same as Ellipsoid::lonLatToXyz */
    int forward(const Vec3& llh, Vec3& xyz) const;

    /** This is same as Ellipsoid::xyzToLonLat */
    int inverse(const Vec3& xyz, Vec3& llh) const;
};

inline void Geocent::print() const
{
    std::cout << "Projection: Geocent" << std::endl
              << "EPSG: " << code() << std::endl;
}

/**
 * UTM coordinate extension of ProjBase
 *
 * EPSG 32601-32660 for Northern Hemisphere
 * EPSG 32701-32760 for Southern Hemisphere
 */
class UTM : public ProjectionBase {
    // Constants related to the projection system
    double lon0;
    int zone;
    bool isnorth;

    // Parameters from Proj.4
    double cgb[6], cbg[6], utg[6], gtu[6];
    double Qn, Zb;

public:
    UTM(int);

    /** @copydoc ProjectionBase::print() */
    void print() const override;

    /** Transform from llh (rad) to UTM (m) */
    int forward(const Vec3& llh, Vec3& xyz) const override;

    /** Transform from UTM(m) to llh (rad) */
    int inverse(const Vec3& xyz, Vec3& llh) const override;
};

inline void UTM::print() const
{
    std::cout << "Projection: UTM" << std::endl
              << "Zone: " << zone << (isnorth ? "N" : "S") << std::endl
              << "EPSG: " << code() << std::endl;
}

/**
 * Polar stereographic extension of ProjBase
 *
 * EPSG: 3413 - Greenland
 * EPSG: 3031 - Antarctica
 */
class PolarStereo : public ProjectionBase {
    // Constants related to projection system
    double lat0, lon0, lat_ts, akm1, e;
    bool isnorth;

public:
    PolarStereo(int);

    /** @copydoc ProjectionBase::print() */
    void print() const override;

    /** Transfrom from llh(rad) to Polar Stereo (m) */
    int forward(const Vec3&, Vec3&) const override;

    /** Transform from Polar Stereo (m) to llh (rad) */
    int inverse(const Vec3&, Vec3&) const override;
};

inline void PolarStereo::print() const
{
    std::cout << "Projection: " << (isnorth ? "North" : "South")
              << " Polar Stereographic" << std::endl
              << "EPSG: " << code() << std::endl;
}

/**
 * Equal Area Projection extension of ProjBase
 *
 * EPSG:6933 for EASE2 grid
 */
class CEA : public ProjectionBase {
    // Constants related to projection system
    double apa[3];
    double lat_ts, k0, e, one_es, qp;

public:
    CEA();

    void print() const override;

    /** Transform from llh (rad) to CEA (m) */
    int forward(const Vec3& llh, Vec3& xyz) const override;

    /** Transform from CEA (m) to LLH (rad) */
    int inverse(const Vec3& xyz, Vec3& llh) const override;
};

inline void CEA::print() const
{
    std::cout << "Projection: Cylindrical Equal Area" << std::endl
              << "EPSG: " << code() << std::endl;
}

// This is to create a projection system from the EPSG code
ProjectionBase* createProj(int epsg);

inline std::unique_ptr<ProjectionBase> makeProjection(int epsg)
{
    return std::unique_ptr<ProjectionBase>(createProj(epsg));
}

// This is to transform a point from one coordinate system to another
int projTransform(ProjectionBase* in, ProjectionBase* out, const Vec3& inpts,
                  Vec3& outpts);

}} // namespace isce3::core
