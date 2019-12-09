
add_subdirectory(msinttypes)
add_subdirectory(internal)
add_subdirectory(error)

# Individual headers
set(EXPORT_HEADERS
    allocators.h
    document.h
    encodedstream.h
    encodings.h
    filereadstream.h
    filewritestream.h
    fwd.h
    istreamwrapper.h
    memorybuffer.h
    memorystream.h
    ostreamwrapper.h
    pointer.h
    prettywriter.h
    rapidjson.h
    reader.h
    schema.h
    stream.h
    stringbuffer.h
    writer.h)

install(FILES ${EXPORT_HEADERS} DESTINATION ${CEREAL_INCLUDEDIR}/external/rapidjson
    COMPONENT cereal_headers)

