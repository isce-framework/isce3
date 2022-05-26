def test_version_sanity():
    from isce3.ext.isce3 import __version__

    assert len(__version__) > 0
