#pragma once

#include <pybind11/pybind11.h>

#include <isce3/core/Projections.h>

// A helper "trampoline" class allowing inheritance in Python by redirecting
// virtual method calls back to the interpreter.
class PyProjectionBase : public isce3::core::ProjectionBase {
    using Base = isce3::core::ProjectionBase;

public:
    using Base::Base;

    void print() const override { PYBIND11_OVERLOAD_PURE(void, Base, print, ); }

    int forward(const isce3::core::Vec3& llh,
                isce3::core::Vec3& xyz) const override
    {
        PYBIND11_OVERLOAD_PURE(int, Base, forward, llh, xyz);
    }

    int inverse(const isce3::core::Vec3& xyz,
                isce3::core::Vec3& llh) const override
    {
        PYBIND11_OVERLOAD_PURE(int, Base, inverse, xyz, llh);
    }
};

void addbinding(
        pybind11::class_<isce3::core::ProjectionBase, PyProjectionBase>&);
void addbinding(pybind11::class_<isce3::core::LonLat>&);
void addbinding(pybind11::class_<isce3::core::Geocent>&);
void addbinding(pybind11::class_<isce3::core::UTM>&);
void addbinding(pybind11::class_<isce3::core::PolarStereo>&);
void addbinding(pybind11::class_<isce3::core::CEA>&);

void addbinding_makeprojection(pybind11::module&);
