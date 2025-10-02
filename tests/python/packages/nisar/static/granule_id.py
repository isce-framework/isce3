from datetime import datetime

from nisar.static.granule_id import form_granule_id


def test_form_granule_id():
    granule_id = form_granule_id(
        mission_id="NISAR",
        radar_band="L",
        product_level=2,
        product_type="STATIC",
        relative_orbit_number=1,
        orbit_pass_direction="ascending",
        frame_number=2,
        x_posting=10.0,
        y_posting=5.0,
        validity_start_datetime=datetime.fromisoformat("1999-12-31T23:59:59"),
        composite_release_id="T01023",
        processing_center="JPL",
        product_counter=1,
    )
    assert (
        granule_id == "NISAR_L2_STATIC_001_A_002_010_005_19991231T235959_T01023_J_001"
    )

    granule_id = form_granule_id(
        mission_id="NISAR",
        radar_band="S",
        product_level=2,
        product_type="STATIC",
        relative_orbit_number=123,
        orbit_pass_direction="descending",
        frame_number=124,
        x_posting=80.0,
        y_posting=80.0,
        validity_start_datetime=datetime.fromisoformat("2000-01-01T00:00:00"),
        composite_release_id="A11111",
        processing_center="somewhere else",
        product_counter=999,
    )
    assert (
        granule_id == "NISAR_S2_STATIC_123_D_124_080_080_20000101T000000_A11111_X_999"
    )
