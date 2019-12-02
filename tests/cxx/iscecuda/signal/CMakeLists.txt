add_isce_test(gpuLooks)

foreach (TESTNAME gpuSignal gpuCrossMul gpuFilter)
    add_isce_test(${TESTNAME})
endforeach ()
