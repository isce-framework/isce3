set(PROGS
crossmul.py
geo2rdr.py
rdr2geo.py
resampSlc.py
)

install(PROGRAMS ${PROGS}
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
