//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2017-2018

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

// isce3::core
#include <isce3/core/DateTime.h>
#include <isce3/core/TimeDelta.h>
#include <isce3/except/Error.h>

TEST(DateTimeTest, StandardConstruction)
{
    isce3::core::DateTime dtime(2017, 5, 12, 1, 12, 30.141592);
    ASSERT_EQ(dtime.year, 2017);
    ASSERT_EQ(dtime.months, 5);
    ASSERT_EQ(dtime.days, 12);
    ASSERT_EQ(dtime.hours, 1);
    ASSERT_EQ(dtime.minutes, 12);
    ASSERT_EQ(dtime.seconds, 30);
    ASSERT_NEAR(dtime.frac, 0.141592, 1.0e-6);
}

TEST(DateTimeTest, NonStandardConstructionV1)
{
    isce3::core::DateTime dtime(2017, 5, 12, 1, 62, 30.141592);
    ASSERT_EQ(dtime.year, 2017);
    ASSERT_EQ(dtime.months, 5);
    ASSERT_EQ(dtime.days, 12);
    ASSERT_EQ(dtime.hours, 2);
    ASSERT_EQ(dtime.minutes, 2);
    ASSERT_EQ(dtime.seconds, 30);
    ASSERT_NEAR(dtime.frac, 0.141592, 1.0e-6);
}

TEST(DateTimeTest, NonStandardConstructionV2)
{
    isce3::core::DateTime dtime(2017, 5, 12, 1, 62, 130.141592);
    ASSERT_EQ(dtime.year, 2017);
    ASSERT_EQ(dtime.months, 5);
    ASSERT_EQ(dtime.days, 12);
    ASSERT_EQ(dtime.hours, 2);
    ASSERT_EQ(dtime.minutes, 4);
    ASSERT_EQ(dtime.seconds, 10);
    ASSERT_NEAR(dtime.frac, 0.141592, 1.0e-6);
}

TEST(DateTimeTest, FromString)
{
    isce3::core::DateTime dtime("2017-05-12T01:12:30.141592");
    ASSERT_EQ(dtime.year, 2017);
    ASSERT_EQ(dtime.months, 5);
    ASSERT_EQ(dtime.days, 12);
    ASSERT_EQ(dtime.hours, 1);
    ASSERT_EQ(dtime.minutes, 12);
    ASSERT_EQ(dtime.seconds, 30);
    ASSERT_NEAR(dtime.frac, 0.141592, 1.0e-6);
    // Test iso string w/o fractional part
    isce3::core::DateTime dtime1("2017-05-12T01:12:30");
    ASSERT_EQ(dtime1.year, 2017);
    ASSERT_EQ(dtime1.months, 5);
    ASSERT_EQ(dtime1.days, 12);
    ASSERT_EQ(dtime1.hours, 1);
    ASSERT_EQ(dtime1.minutes, 12);
    ASSERT_EQ(dtime1.seconds, 30);
    ASSERT_NEAR(dtime1.frac, 0.0, 1.0e-6);

    // Test constructor with other separator
    isce3::core::DateTime dtm("2017-05-12 01:12:30.141592");
    EXPECT_EQ(dtm.year, 2017) << "Wrong year for white-space sep!";
    EXPECT_EQ(dtm.months, 5) << "Wrong month for white-space sep!";
    EXPECT_EQ(dtm.days, 12) << "Wrong day for white-space sep!";
    EXPECT_EQ(dtm.hours, 1) << " Wrong hous for white-space sep!";
    EXPECT_EQ(dtm.minutes, 12) << "Wrong minutes for white-space sep!";
    EXPECT_EQ(dtm.seconds, 30) << "Wrong seconds for white-space sep!";
    EXPECT_FLOAT_EQ(dtm.frac, 0.141592)
            << "Wrong fraction of sec for white-space sep!";

    isce3::core::DateTime dtm1("2017-05-12 01:12:30");
    EXPECT_EQ(dtm1.year, 2017) << "Wrong year for white-space sep!";
    EXPECT_EQ(dtm1.months, 5) << "Wrong month for white-space sep!";
    EXPECT_EQ(dtm1.days, 12) << "Wrong day for white-space sep!";
    EXPECT_EQ(dtm1.hours, 1) << " Wrong hous for white-space sep!";
    EXPECT_EQ(dtm1.minutes, 12) << "Wrong minutes for white-space sep!";
    EXPECT_EQ(dtm1.seconds, 30) << "Wrong seconds for white-space sep!";
    EXPECT_FLOAT_EQ(dtm1.frac, 0.0)
            << "Wrong fraction of sec for white-space sep w/o frac!";

    isce3::core::DateTime dtm3("2017-05-12T01:12:30:141592");
    EXPECT_EQ(dtm3.year, 2017)
            << "Wrong year for T sep and decimal char colon!";
    EXPECT_EQ(dtm3.months, 5)
            << "Wrong month for T sep and decimal char colon!";
    EXPECT_EQ(dtm3.days, 12) << "Wrong day for T sep and decimal char colon!";
    EXPECT_EQ(dtm3.hours, 1) << " Wrong hous for T sep and decimal char colon!";
    EXPECT_EQ(dtm3.minutes, 12)
            << "Wrong minutes for T sep and decimal char colon!";
    EXPECT_EQ(dtm3.seconds, 30)
            << "Wrong seconds for T sep and decimal char colon!";
    EXPECT_FLOAT_EQ(dtm3.frac, 0.141592)
            << "Wrong fraction of sec for T sep and decimal char colon!";

    isce3::core::DateTime dtm4("2017-05-12 01:12:30:141592");
    EXPECT_EQ(dtm4.year, 2017)
            << "Wrong year for space sep and decimal char colon!";
    EXPECT_EQ(dtm4.months, 5)
            << "Wrong month for space sep and decimal char colon!";
    EXPECT_EQ(dtm4.days, 12)
            << "Wrong day for space sep and decimal char colon!";
    EXPECT_EQ(dtm4.hours, 1)
            << " Wrong hous for space sep and decimal char colon!";
    EXPECT_EQ(dtm4.minutes, 12)
            << "Wrong minutes for space sep and decimal char colon!";
    EXPECT_EQ(dtm4.seconds, 30)
            << "Wrong seconds for space sep and decimal char colon!";
    EXPECT_FLOAT_EQ(dtm4.frac, 0.141592)
            << "Wrong fraction of sec for space sep and decimal char colon!";

    isce3::core::DateTime dtm5("2017-05-12T01:12:30,141592");
    EXPECT_EQ(dtm5.year, 2017)
            << "Wrong year for T sep and decimal char comma!";
    EXPECT_EQ(dtm5.months, 5)
            << "Wrong month for T sep and decimal char comma!";
    EXPECT_EQ(dtm5.days, 12) << "Wrong day for T sep and decimal char comma!";
    EXPECT_EQ(dtm5.hours, 1) << " Wrong hous for T sep and decimal char comma!";
    EXPECT_EQ(dtm5.minutes, 12)
            << "Wrong minutes for T sep and decimal char comma!";
    EXPECT_EQ(dtm5.seconds, 30)
            << "Wrong seconds for T sep and decimal char comma!";
    EXPECT_FLOAT_EQ(dtm5.frac, 0.141592)
            << "Wrong fraction of sec for T sep and decimal char comma!";

    isce3::core::DateTime dtm6("2017-05-12T01:12:30");
    EXPECT_EQ(dtm6.year, 2017) << "Wrong year for T sep and w/o frac sec!";
    EXPECT_EQ(dtm6.months, 5) << "Wrong month for T sep and w/o frac sec!";
    EXPECT_EQ(dtm6.days, 12) << "Wrong day for T sep and w/o frac sec!";
    EXPECT_EQ(dtm6.hours, 1) << " Wrong hous for T sep and w/o frac sec!";
    EXPECT_EQ(dtm6.minutes, 12) << "Wrong minutes for T sep and w/o frac sec!";
    EXPECT_EQ(dtm6.seconds, 30) << "Wrong seconds for T sep and w/o frac sec!";
    EXPECT_FLOAT_EQ(dtm6.frac, 0.0)
            << "Wrong fraction of sec for T sep and w/o frac sec!";

    isce3::core::DateTime dtm7("2017-05-12 01:12:30");
    EXPECT_EQ(dtm7.year, 2017) << "Wrong year for space sep and w/o frac sec!";
    EXPECT_EQ(dtm7.months, 5) << "Wrong month for space sep and w/o frac sec!";
    EXPECT_EQ(dtm7.days, 12) << "Wrong day for space sep and w/o frac sec!";
    EXPECT_EQ(dtm7.hours, 1) << " Wrong hous for space sep and w/o frac sec!";
    EXPECT_EQ(dtm7.minutes, 12)
            << "Wrong minutes for space sep and w/o frac sec!";
    EXPECT_EQ(dtm7.seconds, 30)
            << "Wrong seconds for space sep and w/o frac sec!";
    EXPECT_FLOAT_EQ(dtm7.frac, 0.0)
            << "Wrong fraction of sec for space sep and w/o frac sec!";

    isce3::core::DateTime dtm8("2017-05-12");
    EXPECT_EQ(dtm8.year, 2017) << "Wrong year for T sep and w/o time!";
    EXPECT_EQ(dtm8.months, 5) << "Wrong month for T sep and w/o time!";
    EXPECT_EQ(dtm8.days, 12) << "Wrong day for T sep and w/o time!";
    EXPECT_EQ(dtm8.hours, 0) << " Wrong hous for T sep and w/o time!";
    EXPECT_EQ(dtm8.minutes, 0) << "Wrong minutes for T sep and w/o time!";
    EXPECT_EQ(dtm8.seconds, 0) << "Wrong seconds for T sep and w/o time!";
    EXPECT_FLOAT_EQ(dtm8.frac, 0.0)
            << "Wrong fraction of sec for T sep and w/o time!";
}

TEST(DateTimeTest, ThrowExceptFromString)
{
    EXPECT_FALSE(
            isce3::core::DateTime::isIsoFormat("2017-05-12  01:12:30.141592"))
            << "Expect incompatible ISO8601 format!";
    EXPECT_FALSE(
            isce3::core::DateTime::isIsoFormat("2017-05-12t01:12:30.141592"))
            << "Expect incompatible ISO8601 format!";
    EXPECT_TRUE(
            isce3::core::DateTime::isIsoFormat("2017-05-12 01:12:30:141592"))
            << "Expect compatible ISO8601 format with space and :!";
    EXPECT_TRUE(
            isce3::core::DateTime::isIsoFormat("2017-05-12T01:12:30:141592"))
            << "Expect compatible ISO8601 format with T and : !";

    EXPECT_TRUE(
            isce3::core::DateTime::isIsoFormat("2017-05-12T01:12:30,141592"))
            << "Expect compatible ISO8601 format with T and , !";

    EXPECT_TRUE(isce3::core::DateTime::isIsoFormat("2017-05-12T01:12:30"))
            << "Expect compatible ISO8601 format w/o any fraction";

    EXPECT_TRUE(isce3::core::DateTime::isIsoFormat("2017-05-12"))
            << "Expect compatible ISO8601 format with date only!";

    EXPECT_THROW(isce3::core::DateTime("2017-05-12t01:12:30.141592"),
                 isce3::except::InvalidArgument)
            << "Expected wrong ISO8601 format! Sep is 't'";
}

TEST(DateTimeTest, ToString)
{
    isce3::core::DateTime dtime(2017, 5, 12, 1, 12, 30.141592);
    ASSERT_EQ(dtime.isoformat(), "2017-05-12T01:12:30.141592000");
}

TEST(DateTimeTest, BasicTimeDelta)
{
    isce3::core::DateTime dtime(2017, 5, 12, 1, 12, 30.141592);
    dtime += 25.0;
    ASSERT_EQ(dtime.year, 2017);
    ASSERT_EQ(dtime.months, 5);
    ASSERT_EQ(dtime.days, 12);
    ASSERT_EQ(dtime.hours, 1);
    ASSERT_EQ(dtime.minutes, 12);
    ASSERT_EQ(dtime.seconds, 55);
    ASSERT_NEAR(dtime.frac, 0.141592, 1.0e-6);
}

TEST(DateTimeTest, TimeDeltaSub)
{
    isce3::core::DateTime dtime1(2017, 5, 12, 1, 12, 30.141592);
    isce3::core::DateTime dtime2(2017, 5, 13, 2, 12, 33.241592);
    isce3::core::TimeDelta dt = dtime2 - dtime1;
    ASSERT_EQ(dt.days, 1);
    ASSERT_EQ(dt.hours, 1);
    ASSERT_EQ(dt.minutes, 0);
    ASSERT_EQ(dt.seconds, 3);
    ASSERT_NEAR(dt.frac, 0.1, 1.0e-6);
    ASSERT_NEAR(dt.getTotalSeconds(), 90003.1, 1.0e-8);
}

TEST(DateTimeTest, TimeDeltaAdd)
{
    isce3::core::DateTime dtime(2017, 5, 12, 1, 12, 30.141592);
    isce3::core::TimeDelta dt(1, 12, 5.5);
    dtime += dt;
    ASSERT_EQ(dtime.year, 2017);
    ASSERT_EQ(dtime.months, 5);
    ASSERT_EQ(dtime.days, 12);
    ASSERT_EQ(dtime.hours, 2);
    ASSERT_EQ(dtime.minutes, 24);
    ASSERT_EQ(dtime.seconds, 35);
    ASSERT_NEAR(dtime.frac, 0.641592, 1.0e-6);
}

TEST(DateTimeTest, Epoch)
{
    isce3::core::DateTime dtime(2017, 5, 12, 1, 12, 30.141592);
    ASSERT_NEAR(dtime.secondsSinceEpoch(), 1494551550.141592, 1.0e-6);
    dtime.secondsSinceEpoch(1493626353.141592026);
    ASSERT_EQ(dtime.isoformat(), "2017-05-01T08:12:33.141592026");
}

TEST(DateTimeTest, TimeDeltaAssign)
{
    isce3::core::DateTime dtime(2017, 5, 12, 1, 12, 30.141592026);
    isce3::core::TimeDelta dt;
    dt = 3.5;
    dtime += dt;
    ASSERT_EQ(dtime.isoformat(), "2017-05-12T01:12:33.641592026");
}

TEST(DateTimeTest, Comparison)
{
    isce3::core::DateTime dtime1(2017, 5, 12, 1, 12, 30.141592);
    isce3::core::DateTime dtime2(2017, 5, 12, 1, 12, 30.141592);
    isce3::core::DateTime dtime3(2017, 5, 13, 2, 12, 33.241592);
    ASSERT_TRUE(dtime1 == dtime2);
    ASSERT_TRUE(dtime1 != dtime3);
    ASSERT_TRUE(dtime3 > dtime2);
    ASSERT_TRUE(dtime1 <= dtime2);
    ASSERT_TRUE(dtime1 <= dtime3);
}

TEST(DateTimeTest, IsClose)
{
    isce3::core::DateTime dtime1(2017, 5, 12, 1, 12, 30.141592);
    isce3::core::DateTime dtime2 = dtime1 + isce3::core::TimeDelta(1e-11);
    ASSERT_TRUE(dtime1.isClose(dtime2));
    isce3::core::TimeDelta errtol(1e-12);
    ASSERT_FALSE(dtime1.isClose(dtime2, errtol));
}

TEST(DateTimeTest, TimeDeltaComparison)
{
    isce3::core::TimeDelta dt1(0.5);
    isce3::core::TimeDelta dt2(0.5);
    isce3::core::TimeDelta dt3(-0.5);
    ASSERT_TRUE(dt1 == dt2);
    ASSERT_TRUE(dt1 != dt3);
    ASSERT_TRUE(dt3 < dt2);
    ASSERT_TRUE(dt1 >= dt2);
    ASSERT_TRUE(dt1 >= dt3);
}

TEST(DateTimeTest, Normalize)
{
    isce3::core::DateTime t0(1999, 12, 31, 23, 59, 59);
    isce3::core::TimeDelta dt = 1.0;
    isce3::core::DateTime t1 = t0 + dt;

    isce3::core::DateTime expected(2000, 1, 1);

    EXPECT_TRUE(t1.isClose(expected));
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
