set(TESTNAMES
    serializeOrbit
    serializeAttitude
    serializeDoppler
    )

foreach(TESTNAME ${TESTNAMES})
    add_isce_test(${TESTNAME})
endforeach()
