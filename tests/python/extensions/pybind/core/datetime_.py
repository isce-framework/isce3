#!/usr/bin/env python3

import numpy.testing as npt

import isce3.ext.isce3 as isce

def test_default_ctor():
    t = isce.core.DateTime()
    print(t)

def test_from_ordinal():
    t = isce.core.DateTime(730120.)

    assert( t.year == 2000 )
    assert( t.month == 1 )
    assert( t.day == 1 )
    assert( t.hour == 0 )
    assert( t.minute == 0 )
    assert( t.second == 0 )
    npt.assert_almost_equal( t.frac, 0. )

def test_from_ymd():
    year = 2000
    month = 1
    day = 2
    t = isce.core.DateTime(year, month, day)

    assert( t.year == year )
    assert( t.month == month )
    assert( t.day == day )

def test_from_ymdhms():
    year = 2000
    month = 1
    day = 2
    hour = 3
    minute = 4
    second = 5
    t = isce.core.DateTime(year, month, day, hour, minute, second)

    assert( t.year == year )
    assert( t.month == month )
    assert( t.day == day )
    assert( t.hour == hour )
    assert( t.minute == minute )
    assert( t.second == second )

def test_from_ymdhmss():
    year = 2000
    month = 1
    day = 2
    hour = 3
    minute = 4
    second = 5.6
    t = isce.core.DateTime(year, month, day, hour, minute, second)

    assert( t.year == year )
    assert( t.month == month )
    assert( t.day == day )
    assert( t.hour == hour )
    assert( t.minute == minute )
    assert( t.second == 5 )
    npt.assert_almost_equal( t.frac, 0.6 )

def test_from_ymdhmsfrac():
    year = 2000
    month = 1
    day = 2
    hour = 3
    minute = 4
    second = 5
    frac = 0.6
    t = isce.core.DateTime(year, month, day, hour, minute, second, frac)

    assert( t.year == year )
    assert( t.month == month )
    assert( t.day == day )
    assert( t.hour == hour )
    assert( t.minute == minute )
    assert( t.second == second )
    npt.assert_almost_equal( t.frac, 0.6 )

def test_from_string():
    t = isce.core.DateTime("2017-05-12T01:12:30.141592")

    assert( t.year == 2017 )
    assert( t.month == 5 )
    assert( t.day == 12 )
    assert( t.hour == 1 )
    assert( t.minute == 12 )
    assert( t.second == 30 )
    npt.assert_almost_equal( t.frac, 0.141592 )

def test_from_datetime_datetime():
    import datetime
    t1 = datetime.datetime(year=2017, month=5, day=12, hour=1, minute=12, second=30, microsecond=141592)
    t2 = isce.core.DateTime(t1)

    assert( t2.year == 2017 )
    assert( t2.month == 5 )
    assert( t2.day == 12 )
    assert( t2.hour == 1 )
    assert( t2.minute == 12 )
    assert( t2.second == 30 )
    npt.assert_almost_equal( t2.frac, 0.141592 )

def test_comparison():
    t1 = isce.core.DateTime(2000, 1, 1)
    t2 = isce.core.DateTime(2000, 1, 1)
    t3 = isce.core.DateTime(2000, 1, 2)

    assert( t3 > t1 )
    assert( t2 < t3 )
    assert( t1 >= t2 )
    assert( t1 <= t3 )
    assert( t1 == t2 )
    assert( t1 != t3 )

def test_add():
    t1 = isce.core.DateTime(2000, 1, 2, 3, 4, 5, 0.6)
    dt = isce.core.TimeDelta(7, 8, 9, 10, 0.11)

    t2 = t1 + dt

    assert( t2.year == 2000 )
    assert( t2.month == 1 )
    assert( t2.day == 9 )
    assert( t2.hour == 11 )
    assert( t2.minute == 13 )
    assert( t2.second == 15 )
    npt.assert_almost_equal( t2.frac, 0.71 )

    t1 += dt

    assert( t1.year == 2000 )
    assert( t1.month == 1 )
    assert( t1.day == 9 )
    assert( t1.hour == 11 )
    assert( t1.minute == 13 )
    assert( t1.second == 15 )
    npt.assert_almost_equal( t1.frac, 0.71 )

def test_subtract():
    t1 = isce.core.DateTime(2000, 1, 2, 3, 4, 5, 0.6)
    dt = isce.core.TimeDelta(7, 8, 9, 10, 0.11)

    t2 = t1 - dt

    assert( t2.year == 1999 )
    assert( t2.month == 12 )
    assert( t2.day == 25 )
    assert( t2.hour == 18 )
    assert( t2.minute == 54 )
    assert( t2.second == 55 )
    npt.assert_almost_equal( t2.frac, 0.49 )

    t1 -= dt

    assert( t1.year == 1999 )
    assert( t1.month == 12 )
    assert( t1.day == 25 )
    assert( t1.hour == 18 )
    assert( t1.minute == 54 )
    assert( t1.second == 55 )
    npt.assert_almost_equal( t1.frac, 0.49 )

# XXX c++ not implemented
#def test_day_of_year():
#    t = isce.core.DateTime(2001, 2, 3)
#    assert( t.day_of_year() == 34 )

def test_seconds_of_day():
    t = isce.core.DateTime(2000, 1, 2, 3, 4, 5, 0.6)
    npt.assert_almost_equal( t.seconds_of_day(), 11045.6 )

# XXX c++ not implemented
#def test_day_of_week():
#    t = isce.core.DateTime(2001, 2, 3)
#    assert( t.day_of_week() == 6 ) # saturday

# XXX c++ not implemented
#def test_toordinal():
#    t = isce.core.DateTime(2000, 1, 1)
#    npt.assert_almost_equal( t.toordinal(), 730120. )

def test_isoformat():
    s = "2017-05-12T01:12:30.141592000"
    t = isce.core.DateTime(s)
    assert( t.isoformat() == s )

def test_isoformat_usec():
    s = "2017-05-12T01:12:30.141592123"
    t = isce.core.DateTime(s)
    assert( t.isoformat_usec() == s[:26] )

def test_isoformat_roundtrip():
    # DateTime.isoformat() only writes 9 digits, so choose eps < 1e-9
    eps = 5e-10

    # Construct a sequence of DateTimes separated by fixed intervals. The
    # temporal spacing is not an integer number of nanoseconds.
    n = 5
    t0 = isce.core.DateTime(2022, 5, 4, 10, 3, 0.0)
    dt = isce.core.TimeDelta(1.0 + eps)
    times = [t0 + i * dt for i in range(n)]

    # Converts DateTime -> str -> DateTime.
    def roundtrip(t):
        s = t.isoformat()
        return isce.core.DateTime(s)

    # Check that the tolerance of `is_close()` is loose enough to accommodate
    # truncation of DateTimes due to serialization to text.
    for t in times:
        assert t.is_close(roundtrip(t))
