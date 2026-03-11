from nisar.products import build_projection_dataset_attrs_dict


class TestBuildProjectionDatasetAttrsDict:
    def test_epsg4326(self):
        epsg = 4326
        attrs = build_projection_dataset_attrs_dict(epsg)

        # Reference values obtained from https://epsg.io/4326.
        assert attrs.pop("spatial_ref").startswith(b'GEOGCS["WGS 84",')
        assert attrs == dict(
            epsg_code=epsg,
            semi_major_axis=6378137.0,
            inverse_flattening=298.257223563,
            ellipsoid=b"WGS84",
            grid_mapping_name=b"latitude_longitude",
            longitude_of_prime_meridian=0.0,
        )

    def test_epsg3413(self):
        epsg = 3413
        attrs = build_projection_dataset_attrs_dict(epsg)

        # Reference values obtained from https://epsg.io/3413.
        assert attrs.pop("spatial_ref").startswith(
            b'PROJCS["WGS 84 / NSIDC Sea Ice Polar Stereographic North",'
        )
        assert attrs == dict(
            epsg_code=epsg,
            semi_major_axis=6378137.0,
            inverse_flattening=298.257223563,
            ellipsoid=b"WGS84",
            grid_mapping_name=b"polar_stereographic",
            false_easting=0.0,
            false_northing=0.0,
            longitude_of_projection_origin=0.0,
            latitude_of_projection_origin=90.0,
            standard_parallel=70.0,
            straight_vertical_longitude_from_pole=-45.0,
        )

    def test_epsg32601(self):
        epsg = 32601
        attrs = build_projection_dataset_attrs_dict(epsg)

        # Reference values obtained from https://epsg.io/32601.
        assert attrs.pop("spatial_ref").startswith(b'PROJCS["WGS 84 / UTM zone 1N",')
        assert attrs == dict(
            epsg_code=epsg,
            semi_major_axis=6378137.0,
            inverse_flattening=298.257223563,
            ellipsoid=b"WGS84",
            grid_mapping_name=b"transverse_mercator",
            false_easting=500_000.0,
            false_northing=0.0,
            longitude_of_projection_origin=0.0,
            latitude_of_projection_origin=0.0,
            utm_zone_number=1,
            longitude_of_central_meridian=-177.0,
            scale_factor_at_central_meridian=0.9996,
        )

    def test_epsg6933(self):
        epsg = 6933
        attrs = build_projection_dataset_attrs_dict(epsg)

        # Reference values obtained from https://epsg.io/6933.
        assert attrs.pop("spatial_ref").startswith(
            b'PROJCS["WGS 84 / NSIDC EASE-Grid 2.0 Global",'
        )
        assert attrs == dict(
            epsg_code=epsg,
            semi_major_axis=6378137.0,
            inverse_flattening=298.257223563,
            ellipsoid=b"WGS84",
            grid_mapping_name=b"cylindrical_equal_area",
            false_easting=0.0,
            false_northing=0.0,
            longitude_of_projection_origin=0.0,
            latitude_of_projection_origin=0.0,
            longitude_of_central_meridian=0.0,
            standard_parallel=30.0,
        )
