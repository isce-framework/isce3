add_subdirectory(internal)

set (EXPORT_HEADERS     gtest.h
                        gtest-death-test.h
                        gtest-message.h
                        gtest-param-test.h
                        gtest-printers.h
                        gtest-spi.h
                        gtest-test-part.h
                        gtest-typed-test.h
                        gtest_pred_impl.h
                        gtest_prod.h )

install(FILES           ${EXPORT_HEADERS}
        DESTINATION     ${GTEST_INCLUDEDIR}
        COMPONENT       gtest_headers)


