foreach (TESTNAME ih5
                  ih5gdal 
                  ih5nativeread 
                  ih5nativewrite 
                  ih5castread 
                  ih5castwrite)
        add_isce_test(${TESTNAME})
endforeach ()
