set(PROGS
crossmul.py
geo2rdr.py
rdr2geo.py
resampSlc.py
)

install(PROGRAMS ${PROGS}
        DESTINATION "${ISCE_BINDIR}"
        )
