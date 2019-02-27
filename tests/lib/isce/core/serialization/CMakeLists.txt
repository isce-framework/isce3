foreach (TESTNAME serializeOrbit serializeAttitude serializeDoppler)
    add_isce_test(${TESTNAME} ${ISCE_SOURCE_DIR}/contrib/cereal/include)
endforeach ()
