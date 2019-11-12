add_isce_test(gpuLooks)

foreach (TESTNAME gpuSignal gpuCrossMul gpuFilter)
    add_isce_test(${TESTNAME})
    add_dependencies(${TESTNAME} geom_test_data)
endforeach ()
