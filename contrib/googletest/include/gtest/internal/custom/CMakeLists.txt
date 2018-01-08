set ( EXPORT_HEADERS    gtest-port.h
                        gtest-printers.h
                        gtest.h )

install(FILES       ${EXPORT_HEADERS}
        DESTINATION ${GTEST_INCLUDEDIR}/internal/custom
        COMPONENT   gtest_headers)
