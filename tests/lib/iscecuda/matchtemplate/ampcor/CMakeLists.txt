foreach (TESTNAME addTiles
                  detect
                  nudge
                  refStats
                  tgtStats
                  maxcor
                  pushTiles  
                  sanity
                  zoomcor
                  correlate
                  migrate
                  refine
                  sat)
    add_isce_test(${TESTNAME})
endforeach ()
