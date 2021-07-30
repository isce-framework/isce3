import iscetest
import numpy.testing as npt
from pathlib import Path
import nisar
import isce3


def test_load_attitude():
    fn = Path(iscetest.data) / "attitude.xml"
    attitude = nisar.products.readers.load_attitude_from_xml(fn)

    # Some spot checks based on visual inspection of the XML file.
    t0 = isce3.core.DateTime(2020, 1, 7, 20, 0, 0)
    assert attitude.reference_epoch == t0
    q = attitude.quaternions[0]
    npt.assert_almost_equal(q.w, 0.45998636868613685)
    npt.assert_almost_equal(q.x, 0.8484777808560364)
    npt.assert_almost_equal(q.y, 0.2604469244480131)
    npt.assert_almost_equal(q.z, -0.02579526239696106)

    # By default first time stamp should be zero.
    assert attitude.time[0] == 0.0

    # Now try reading with a specified epoch.
    epoch = t0 + isce3.core.TimeDelta(seconds=1.0)
    attitude = nisar.products.readers.load_attitude_from_xml(fn, epoch)
    assert attitude.reference_epoch == epoch
    assert attitude.time[0] == -1.0

    # Try to read non-existent S-band data.
    with npt.assert_raises(IOError):
        nisar.products.readers.load_attitude_from_xml(fn, epoch, band="S")
