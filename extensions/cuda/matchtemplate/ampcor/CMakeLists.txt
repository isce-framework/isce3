set(HEADERS
    correlators.h
    kernels.h
    public.h
    Sequential.h
    Sequential.icc)

set(SRCS
        c2r.cu
        detect.cu
        migrate.cu
        offsets.cu
        refStats.cu
        tgtStats.cu
        correlate.cu
        maxcor.cu
        nudge.cu
        r2c.cu
        sat.cu)

# end of file
add_isce_libdir(matchtemplate/ampcor "${SRCS}" "${HEADERS}")
