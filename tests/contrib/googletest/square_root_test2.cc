#include <limits>
#include <cmath>
#include "gtest/gtest.h"


double square_root(const double s){
    // Use Newton's method for zero of f(x) = x**2 - s, f'(x) = 2x
    // 0 = f(x_n) + f'(x_n)*(x_{n+1} - x_n)
    // x_{n+1} = x_n - f(x_n)/f'(x_n)
    // x_{n+1} = 0.5*(x_n + s/x_n)

    // define the precision
    double error = 1.e-12;

    if( s >= 0.0 ){
        // the square root value to be returned initialized to s
        double x = s;
        // loop until precision is satisfied
        while ((x - s/x)/2.0 > error){
            x = (x + s/x) / 2.0;
        }
        return x;
    } else {
        return std::numeric_limits<double>::quiet_NaN();
    }
}



// Struct to count detected failures

struct SquareRootTest : public ::testing::Test {
    virtual void SetUp() {
        fails = 0;
    }
    virtual void TearDown() {
        if (fails > 0) {
            std::cerr << "SquareRoot::TearDown sees failures" << std::endl;
        }
    }
    unsigned fails;
};

// Add a listener to be more quiet on the output
// https://github.com/google/googletest/blob/master/googletest/docs/
// AdvancedGuide.md#extending-google-test-by-handling-test-events

class MinimalistPrinter : public ::testing::EmptyTestEventListener {
    // Called before a test starts.
    virtual void OnTestStart(const ::testing::TestInfo& test_info) {
    // Comment these lines if you want completely silent run
//        printf("*** Test %s.%s.\n",
//               test_info.test_case_name(), test_info.name()
//          );
    }

    // Called after a failed assertion or a SUCCEED() invocation.
    virtual void OnTestPartResult(
        const ::testing::TestPartResult& test_part_result) {
        printf("%s in %s:%d\nSee xml report for details.\n, %s\n",
               test_part_result.failed() ? "*** Failure" : "Success",
               test_part_result.file_name(),
               test_part_result.line_number(),
               test_part_result.summary()
        );
    }

    // Called after a test ends.
    virtual void OnTestEnd(const ::testing::TestInfo& test_info) {
// Uncomment these lines if you want verbose output
//        printf("*** Test %s.%s ending.\n",
//            test_info.test_case_name(), test_info.name()
//        );
    }
};

// Use TEST_F to use the failure counting struct
TEST_F(SquareRootTest, PositiveDoubles) {
    EXPECT_EQ(1., square_root(1.));
    EXPECT_EQ(1., square_root(1.0));
    EXPECT_EQ(18., square_root(324.));
    EXPECT_EQ(25.4, square_root(645.16));
    EXPECT_EQ(50.3321, square_root(2533.32029041));
    fails += ::testing::Test::HasFailure();
}

TEST_F(SquareRootTest, CloseEnoughDoubles) {
    ASSERT_NEAR(50.3321, square_root(2533.32029041), 1.e-12);
    //This one is designed to fail
//    ASSERT_NEAR(50.3321, square_root(2533.320), 1.e-12);
    fails += ::testing::Test::HasFailure();
}

TEST_F(SquareRootTest, ZeroDoubles) {
    ASSERT_EQ(0.0, square_root(0.0));
    fails += ::testing::Test::HasFailure();
}

TEST_F(SquareRootTest, NegativeDoubles) {
    ASSERT_EQ(1, std::isnan(square_root(-22.0)));
    fails += ::testing::Test::HasFailure();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    // get event listeners
    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    // add a listener to the end
    listeners.Append(new MinimalistPrinter );
    // delete the default listener
    delete listeners.Release(listeners.default_result_printer());
    // run the tests
    return RUN_ALL_TESTS();
}
