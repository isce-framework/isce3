#include <gtest/gtest.h>

#include <isce3/core/Utilities.h>

using isce3::core::binarySearch;

TEST(BinarySearchTest, SingleElementArray) {
    std::valarray<double> arr = {2.0};
    EXPECT_EQ(binarySearch(arr, 0.0), 0);
    EXPECT_EQ(binarySearch(arr, 100.0), 0);
}

TEST(BinarySearchTest, SingleElementArrayPickLeft) {
    const bool always_pick_left = true;
    std::valarray<double> arr = {2.0};
    EXPECT_EQ(binarySearch(arr, 0.0, always_pick_left), 0);
    EXPECT_EQ(binarySearch(arr, 100.0, always_pick_left), 0);
}


TEST(BinarySearchTest, ClosestMatchExact) {
    std::valarray<double> arr = {1.0, 3.0, 5.0, 7.0};
    EXPECT_EQ(binarySearch(arr, 3.0), 1);
    EXPECT_EQ(binarySearch(arr, 7.0), 3);
}

TEST(BinarySearchTest, ClosestMatchExactPickLeft) {
    const bool always_pick_left = true;
    std::valarray<double> arr = {1.0, 3.0, 5.0, 7.0};
    EXPECT_EQ(binarySearch(arr, 3.0, always_pick_left), 1);
    EXPECT_EQ(binarySearch(arr, 7.0, always_pick_left), 3);
}

TEST(BinarySearchTest, ClosestMatchInBetween) {
    std::valarray<double> arr = {1.0, 3.0, 5.0, 7.0};
    EXPECT_EQ(binarySearch(arr, 2.1), 1); // closer to `3`
    EXPECT_EQ(binarySearch(arr, 4.1), 2); // closer to `5`
    EXPECT_EQ(binarySearch(arr, 6.4), 3); // closer to `7`
}

TEST(BinarySearchTest, ClosestMatchInBetweenPickLeft) {
    const bool always_pick_left = true;
    std::valarray<double> arr = {1.0, 3.0, 5.0, 7.0};
    EXPECT_EQ(binarySearch(arr, 2.1, always_pick_left), 0);
    EXPECT_EQ(binarySearch(arr, 4.1, always_pick_left), 1);
    EXPECT_EQ(binarySearch(arr, 6.4, always_pick_left), 2);
}

TEST(BinarySearchTest, ValueBeforeFirstElement) {
    std::valarray<double> arr = {1.0, 2.0, 3.0};
    EXPECT_EQ(binarySearch(arr, -1.0), 0); // `-1` closest to `1`
}

TEST(BinarySearchTest, ValueBeforeFirstElementPickLeft) {
    const bool always_pick_left = true;
    std::valarray<double> arr = {1.0, 2.0, 3.0};
    EXPECT_EQ(binarySearch(arr, -1.0, always_pick_left), 0);
}

TEST(BinarySearchTest, ValueAfterLastElement) {
    std::valarray<double> arr = {1.0, 2.0, 3.0};
    EXPECT_EQ(binarySearch(arr, 5.0), 2); // `5` closest to `3`
}

TEST(BinarySearchTest, ValueAfterLastElementPickLeft) {
    const bool always_pick_left = true;
    std::valarray<double> arr = {1.0, 2.0, 3.0};
    EXPECT_EQ(binarySearch(arr, 5.0, always_pick_left), 2);
}

TEST(BinarySearchTest, EqualDistanceChoosesLeft) {
    std::valarray<double> arr = {1.0, 2.0, 3.0, 4.0};

    // `2` and `3` equally distant; picks left (`2`)
    EXPECT_EQ(binarySearch(arr, 2.5), 1);
}

TEST(BinarySearchTest, EqualDistanceChoosesLeftPickLeft) {
    const bool always_pick_left = true;
    std::valarray<double> arr = {1.0, 2.0, 3.0, 4.0};

    // `2` and `3` equally distant; picks left (`2`)
    EXPECT_EQ(binarySearch(arr, 2.5, always_pick_left), 1);
}

TEST(BinarySearchTest, ThrowsOnEmptyArray) {
    std::valarray<double> arr;
    EXPECT_THROW(binarySearch(arr, 0.0), std::invalid_argument);
}

TEST(BinarySearchTest, ThrowsOnNaN) {
    std::valarray<double> arr = {1.0, 2.0, NAN, 4.0};
    EXPECT_THROW(binarySearch(arr, 3.0), std::invalid_argument);
}

int main(int argc, char * argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
