
# Individual headers
set(EXPORT_HEADERS
    biginteger.h
    diyfp.h
    dtoa.h
    ieee754.h
    itoa.h
    meta.h
    pow10.h
    regex.h
    stack.h
    strfunc.h
    strtod.h
    swap.h)

install(FILES ${EXPORT_HEADERS} DESTINATION ${CEREAL_INCLUDEDIR}/external/rapidjson/internal
    COMPONENT cereal_headers)

