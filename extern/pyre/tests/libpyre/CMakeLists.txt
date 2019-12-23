set(testfiles
algebra/bcd.cc
geometry/brick.cc
geometry/point.cc
geometry/pointcloud.cc
grid/grid-direct-data.cc
grid/grid-direct-get.cc
grid/grid-direct-set.cc
grid/grid-direct.cc
grid/grid-fill.cc
grid/grid-heap.cc
grid/grid-mosaic.cc
grid/grid-scan.cc
grid/grid-transform.cc
grid/grid-view-assignment.cc
grid/grid-view.cc
grid/index-access.cc
grid/index-arithmetic.cc
grid/index-bool.cc
grid/index.cc
grid/iterator-access.cc
grid/iterator-loop.cc
grid/iterator-slice.cc
grid/iterator.cc
grid/layout-order-default.cc
grid/layout-order.cc
grid/layout-slice.cc
grid/layout.cc
grid/packing-access.cc
grid/packing-c.cc
grid/packing-fortran.cc
grid/packing.cc
grid/slice.cc
journal/channel.cc
journal/chronicler.cc
journal/debug-envvar.cc
journal/debug-injection.cc
journal/debug-null.cc
journal/debug.cc
journal/diagnostic-injection.cc
journal/diagnostic.cc
journal/error-injection.cc
journal/error.cc
journal/firewall-injection.cc
journal/firewall-null.cc
journal/firewall.cc
journal/index-inventory.cc
journal/index.cc
journal/info-injection.cc
journal/info.cc
journal/inventory.cc
journal/sanity.cc
journal/warning-injection.cc
journal/warning.cc
# memory/constdirect-clone.cc XXX
memory/constdirect-create.cc
memory/constdirect-instantiate-partial.cc
memory/constdirect-instantiate.cc
memory/constdirect-map.cc
memory/constview-instantiate.cc
# memory/direct-clone.cc XXX
memory/direct-create.cc
memory/direct-instantiate-partial.cc
memory/direct-instantiate.cc
memory/direct-map.cc
memory/heap-instantiate.cc
memory/view-instantiate.cc
timers/timer.cc
)

if(MPI_FOUND)
    list(APPEND testfiles
        mpi/communicator.cc
        mpi/group-exclude.cc
        mpi/group-include.cc
        mpi/group-setops.cc
        mpi/group.cc
        mpi/sanity.cc
        mpi/world.cc
        )
endif()

foreach(testfile ${testfiles})
    # generate target and test names from file/directory
    get_filename_component(name ${testfile} NAME_WE)
    get_filename_component(dir  ${testfile} DIRECTORY)
    string(REPLACE "/" "." target test.libpyre.${dir}.${name})
    string(REPLACE "/" "." testname    libpyre.${dir}.${name})

    add_executable(${target} ${testfile})
    target_link_libraries(${target} PUBLIC pyre)
    add_test(NAME ${testname} COMMAND ${target})
endforeach()
