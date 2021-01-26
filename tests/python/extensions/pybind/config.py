def test_version_sanity():
    from pybind_isce3 import __version__

    assert len(__version__) > 0
