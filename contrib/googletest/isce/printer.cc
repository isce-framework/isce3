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
