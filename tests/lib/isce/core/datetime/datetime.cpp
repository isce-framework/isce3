//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2017-2018

#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include "gtest/gtest.h"

// isce::core
#include "isce/core/DateTime.h"

TEST(DateTimeTest, StandardConstruction) {
    isce::core::DateTime dtime(2017, 5, 12, 1, 12, 30.141592);
    ASSERT_EQ(dtime.year, 2017);
    ASSERT_EQ(dtime.months, 5);
    ASSERT_EQ(dtime.days, 12);
    ASSERT_EQ(dtime.hours, 1);
    ASSERT_EQ(dtime.minutes, 12);
    ASSERT_EQ(dtime.seconds, 30);
    ASSERT_NEAR(dtime.frac, 0.141592, 1.0e-6);
}

TEST(DateTimeTest, NonStandardConstructionV1) {
    isce::core::DateTime dtime(2017, 5, 12, 1, 62, 30.141592);
    ASSERT_EQ(dtime.year, 2017);
    ASSERT_EQ(dtime.months, 5);
    ASSERT_EQ(dtime.days, 12);
    ASSERT_EQ(dtime.hours, 2);
    ASSERT_EQ(dtime.minutes, 2);
    ASSERT_EQ(dtime.seconds, 30);
    ASSERT_NEAR(dtime.frac, 0.141592, 1.0e-6);
}

TEST(DateTimeTest, NonStandardConstructionV2) {
    isce::core::DateTime dtime(2017, 5, 12, 1, 62, 130.141592);
    ASSERT_EQ(dtime.year, 2017);
    ASSERT_EQ(dtime.months, 5);
    ASSERT_EQ(dtime.days, 12);
    ASSERT_EQ(dtime.hours, 2);
    ASSERT_EQ(dtime.minutes, 4);
    ASSERT_EQ(dtime.seconds, 10);
    ASSERT_NEAR(dtime.frac, 0.141592, 1.0e-6);
}

TEST(DateTimeTest, FromString) {
    isce::core::DateTime dtime("2017-05-12T01:12:30.141592");
    ASSERT_EQ(dtime.year, 2017);
    ASSERT_EQ(dtime.months, 5);
    ASSERT_EQ(dtime.days, 12);
    ASSERT_EQ(dtime.hours, 1);
    ASSERT_EQ(dtime.minutes, 12);
    ASSERT_EQ(dtime.seconds, 30);
    ASSERT_NEAR(dtime.frac, 0.141592, 1.0e-6);
    // Test assignment
    isce::core::DateTime dtime2;
    dtime2 = "2017-05-12T01:12:30.141592";
    ASSERT_EQ(dtime2.year, 2017);
    ASSERT_EQ(dtime2.months, 5);
    ASSERT_EQ(dtime2.days, 12);
    ASSERT_EQ(dtime2.hours, 1);
    ASSERT_EQ(dtime2.minutes, 12);
    ASSERT_EQ(dtime2.seconds, 30);
    ASSERT_NEAR(dtime2.frac, 0.141592, 1.0e-6);
}

TEST(DateTimeTest, ToString) {
    isce::core::DateTime dtime(2017, 5, 12, 1, 12, 30.141592);
    ASSERT_EQ(dtime.isoformat(), "2017-05-12T01:12:30.141592000");
}

TEST(DateTimeTest, BasicTimeDelta) {
    isce::core::DateTime dtime(2017, 5, 12, 1, 12, 30.141592);
    dtime += 25.0;
    ASSERT_EQ(dtime.year, 2017);
    ASSERT_EQ(dtime.months, 5);
    ASSERT_EQ(dtime.days, 12);
    ASSERT_EQ(dtime.hours, 1);
    ASSERT_EQ(dtime.minutes, 12);
    ASSERT_EQ(dtime.seconds, 55);
    ASSERT_NEAR(dtime.frac, 0.141592, 1.0e-6);
}

TEST(DateTimeTest, TimeDeltaSub) {
    isce::core::DateTime dtime1(2017, 5, 12, 1, 12, 30.141592);
    isce::core::DateTime dtime2(2017, 5, 13, 2, 12, 33.241592);
    isce::core::TimeDelta dt = dtime2 - dtime1;
    ASSERT_EQ(dt.days, 1);
    ASSERT_EQ(dt.hours, 1);
    ASSERT_EQ(dt.minutes, 0);
    ASSERT_EQ(dt.seconds, 3);
    ASSERT_NEAR(dt.frac, 0.1, 1.0e-6);
    ASSERT_NEAR(dt.getTotalSeconds(), 90003.1, 1.0e-8);
}

TEST(DateTimeTest, TimeDeltaAdd) {
    isce::core::DateTime dtime(2017, 5, 12, 1, 12, 30.141592);
    isce::core::TimeDelta dt(1, 12, 5.5);
    dtime += dt;
    ASSERT_EQ(dtime.year, 2017);
    ASSERT_EQ(dtime.months, 5);
    ASSERT_EQ(dtime.days, 12);
    ASSERT_EQ(dtime.hours, 2);
    ASSERT_EQ(dtime.minutes, 24);
    ASSERT_EQ(dtime.seconds, 35);
    ASSERT_NEAR(dtime.frac, 0.641592, 1.0e-6);
}

TEST(DateTimeTest, Epoch) {
    isce::core::DateTime dtime(2017, 5, 12, 1, 12, 30.141592);
    ASSERT_NEAR(dtime.secondsSinceEpoch(), 1494551550.141592, 1.0e-6);
    dtime.secondsSinceEpoch(1493626353.141592026);
    ASSERT_EQ(dtime.isoformat(), "2017-05-01T08:12:33.141592026");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
