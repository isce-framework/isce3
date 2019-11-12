foreach(file
    debuginfo.h
    firewalls.h
    macros.h
    manipulators.h manipulators.icc
    public.h
    Channel.h Channel.icc
    Chronicler.h Chronicler.icc
    Console.h
    Debug.h Debug.icc
    Device.h Device.icc
    Diagnostic.h Diagnostic.icc
    Error.h Error.icc
    Firewall.h Firewall.icc
    Index.h Index.icc
    Informational.h Informational.icc
    Inventory.h Inventory.icc
    Locator.h Locator.icc
    Null.h Null.icc
    Renderer.h Renderer.icc
    Selector.h Selector.icc
    Streaming.h Streaming.icc
    Warning.h Warning.icc
    )

    configure_file(${file} ${buildinclude}/pyre/journal/${file} COPYONLY)
endforeach()

configure_file(journal.h ${buildinclude}/pyre/journal.h COPYONLY)

target_sources(pyre PRIVATE
    debuginfo.cc
    firewalls.cc
    journal.cc
    Chronicler.cc
    Console.cc
    Device.cc
    Renderer.cc
    Streaming.cc
    )
