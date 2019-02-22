foreach (TESTNAME serializeEllipsoid serializeOrbit serializeMetadata serializeDoppler)
    add_isce_test(${TESTNAME} ${ISCE_SOURCE_DIR}/contrib/cereal/include)
endforeach ()
