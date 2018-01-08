add_subdirectory(custom)

set ( EXPORT_HEADERS    gtest-death-test-internal.h
                        gtest-filepath.h
                        gtest-internal.h
                        gtest-linked_ptr.h
                        gtest-param-util-generated.h
                        gtest-param-util.h
                        gtest-port-arch.h
                        gtest-port.h
                        gtest-string.h
                        gtest-tuple.h
                        gtest-type-util.h )

install (FILES          ${EXPORT_HEADERS}
         DESTINATION    ${GTEST_INCLUDEDIR}/internal
         COMPONENT      gtest_headers)

