add_isce_test(geo2rdr)
add_dependencies(geo2rdr geom_test_data)
set_tests_properties(geo2rdr PROPERTIES DEPENDS topo)
