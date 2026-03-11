// -*- C++ -*-
//-*- coding: utf-8 -*-
//
// Test code to verify the overridden Merge method in BoundingBox class
//
#include <gtest/gtest.h>
#include <isce3/geometry/Shapes.h>

using namespace isce3::geometry;


BoundingBox get_bbox(double minX, double maxX, double minY, double maxY) {
    BoundingBox bbox;
    bbox.MinX = minX;
    bbox.MaxX = maxX;
    bbox.MinY = minY;
    bbox.MaxY = maxY;
    return bbox;
}


TEST(BboxTest, ordinary_merge_1) {
    // Test the merge method implemented in the base class of BoundingBox
    BoundingBox bbox_4326_1 = get_bbox(-10.0, 20.0, -5.0, 5.0);
    BoundingBox bbox_4326_2 = get_bbox(15.0, 30.0, 0.0, 10.0);

    bbox_4326_1.Merge(bbox_4326_2);

    EXPECT_DOUBLE_EQ(bbox_4326_1.MinX, -10.0);
    EXPECT_DOUBLE_EQ(bbox_4326_1.MaxX, 30.0);
}


TEST(BboxTest, ordinary_merg_2) {
    // Simple merge in geographic coordinates that does not cross the antimeridian
    BoundingBox bbox_4326_1 = get_bbox(-10.0, 20.0, -5.0, 5.0);
    BoundingBox bbox_4326_2 = get_bbox(15.0, 30.0, 0.0, 10.0);

    bbox_4326_1.Merge(bbox_4326_2, 4326);

    EXPECT_DOUBLE_EQ(bbox_4326_1.MinX, -10.0);
    EXPECT_DOUBLE_EQ(bbox_4326_1.MaxX, 30.0);
}


TEST(BboxTest, ordinary_merge_3) {
    BoundingBox bbox_4326_1 = get_bbox(-3.0, 3.0, -5.0, 5.0);
    BoundingBox bbox_4326_2 = get_bbox(-2.0, 5.0, 0.0, 10.0);

    bbox_4326_1.Merge(bbox_4326_2, 4326);

    EXPECT_DOUBLE_EQ(bbox_4326_1.MinX, -3.0);
    EXPECT_DOUBLE_EQ(bbox_4326_1.MaxX, 5.0);
}


TEST(BboxTest, antimeridian_merge) {
    // Simple merge in geographic coordinates that does cross the antimeridian
    BoundingBox bbox_4326_1 = get_bbox(-179.0, -177.0, -5.0, 5.0);
    BoundingBox bbox_4326_2 = get_bbox(179.0, 182.0, 0.0, 10.0);

    bbox_4326_1.Merge(bbox_4326_2, 4326);

    EXPECT_DOUBLE_EQ(bbox_4326_1.MinX, 179.0);
    EXPECT_DOUBLE_EQ(bbox_4326_1.MaxX, 183.0);
}


TEST(BboxTest, antimeridian_merge_2) {
    // Simple merge in geographic coordinates that does cross the antimeridian
    BoundingBox bbox_4326_1 = get_bbox(-177.0, 179.0, -5.0, 5.0);
    BoundingBox bbox_4326_2 = get_bbox(179.0, 182.0, 0.0, 10.0);

    bbox_4326_1.Merge(bbox_4326_2, 4326);

    EXPECT_DOUBLE_EQ(bbox_4326_1.MinX, 179.0);
    EXPECT_DOUBLE_EQ(bbox_4326_1.MaxX, 183.0);
}



TEST(BboxTest, utm_merge) {
    // Merge the bounding box in UTM coordinates.
    // Antimeridian handling is not expected to be triggered.
    BoundingBox bbox_utm_1 = get_bbox(-179.0, -177.0, -5.0, 5.0);
    BoundingBox bbox_utm_2 = get_bbox(179.0, 182.0, 0.0, 10.0);

    bbox_utm_1.Merge(bbox_utm_2, 32601);

    EXPECT_DOUBLE_EQ(bbox_utm_1.MinX, -179.0);
    EXPECT_DOUBLE_EQ(bbox_utm_1.MaxX, 182.0);
}


int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file