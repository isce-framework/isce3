foreach (TESTNAME serializeProductMetadata serializeProduct)
    add_isce_test(${TESTNAME})
    add_dependencies(${TESTNAME} geom_test_data)
endforeach ()
