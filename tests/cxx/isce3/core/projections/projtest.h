#pragma once

#include <iostream>

#include <gtest/gtest.h>

#include <isce3/core/Projections.h>
#include <isce3/core/Vector.h>

using isce3::core::ProjectionBase;
using isce3::core::Vec3;

auto projTest(const ProjectionBase& p, const Vec3& ref_llh, const Vec3& ref_xyz)
{
    Vec3 xyz = p.forward(ref_llh);
    EXPECT_NEAR(xyz[0], ref_xyz[0], 1e-6);
    EXPECT_NEAR(xyz[1], ref_xyz[1], 1e-6);
    EXPECT_NEAR(xyz[2], ref_xyz[2], 1e-6);

    Vec3 llh = p.inverse(ref_xyz);
    EXPECT_NEAR(llh[0], ref_llh[0], 1e-9);
    EXPECT_NEAR(llh[1], ref_llh[1], 1e-9);
    EXPECT_NEAR(llh[2], ref_llh[2], 1e-6);
}

#define PROJ_TEST(testclass, proj, name, ...)                                  \
    TEST_F(testclass, name)                                                    \
    {                                                                          \
        projTest(proj, __VA_ARGS__);                                           \
        fails += ::testing::Test::HasFailure();                                \
    }                                                                          \
    struct consume_semicolon
