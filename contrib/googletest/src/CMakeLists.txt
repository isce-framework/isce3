set (PROJECT_SRCS   gtest.cc
                    gtest-death-test.cc
                    gtest-filepath.cc
                    gtest-matchers.cc
                    gtest-port.cc
                    gtest-printers.cc
                    gtest-test-part.cc
                    gtest-typed-test.cc )

add_library(gtest SHARED
                  ${PROJECT_SRCS})
target_include_directories(gtest PUBLIC .. ../include)
target_link_libraries(gtest pthread)


install(TARGETS gtest
        DESTINATION ${GTEST_LIBDIR}
        COMPONENT gtest_libs)
