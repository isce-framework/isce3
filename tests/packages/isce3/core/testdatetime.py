#!/usr/bin/env python3
import isce3

def testConstruction():
    import datetime

    #Reference datetime
    refobjs = [datetime.datetime(2017,5,12,1,12,30,141592),
               datetime.datetime(2011,12,31,0,23,59,999999),
               datetime.datetime(2019,1,1,0,0,0,1)]
               
   
    for refobj in refobjs:
        #Create using string
        pyobj = isce3.core.dateTime(dt=refobj.isoformat())

        #Create datetime from DateTime
        reconsobj = datetime.datetime.strptime(pyobj.isoformat()[:-3], "%Y-%m-%dT%H:%M:%S.%f")
        assert abs((reconsobj - refobj).total_seconds()) < 1.0e-6

        #Create using datetime
        pyobj = isce3.core.dateTime(dt=refobj)

        #Create datetime from DateTime
        reconsobj = datetime.datetime.strptime(pyobj.isoformat()[:-3], "%Y-%m-%dT%H:%M:%S.%f")

        assert abs((reconsobj - refobj).total_seconds()) < 1.0e-6

    return

def testComparison():
    from isce3.core import DateTime

    dtime1 = isce3.core.dateTime(dt="2017-05-12T01:12:30.141592")
    dtime2 = isce3.core.dateTime(dt="2017-05-13T02:12:33.241592")
   
    ###Copy from dtime1
    dtime3 = isce3.core.dateTime()
    dtime3.set(dtime1.isoformat())

    assert (dtime1 < dtime2)
    assert (dtime2 > dtime1)
    assert (dtime1 >= dtime3)
    assert (dtime1 <= dtime3)
    assert (dtime1 == dtime3)
    assert (dtime1 != dtime2)

    return

def testTimeDelta():

    dtime1 = isce3.core.dateTime(dt="2017-05-12T01:12:30.141592")
    dtime2 = isce3.core.dateTime(dt="2017-05-13T02:12:33.241592")
    delta = dtime2 - dtime1

    assert abs(delta.getTotalSeconds() - 90003.1) < 1.0e-8

    return

def testAddition():
    from isce3.core.DateTime import DateTime 
    from isce3.core.TimeDelta import TimeDelta
    dtime1 = DateTime(dt="2017-05-12T01:12:30.141592")
    delta = TimeDelta(dt=3.0)

    dtime2 = dtime1 + delta
    assert(dtime2.isoformat() == "2017-05-12T01:12:33.141592000")

    delta = TimeDelta(-4.0)
    dtime2 = dtime1 + delta
    assert(dtime2.isoformat() == "2017-05-12T01:12:26.141592000")

    return

def testUnits():

    for value in [90456.767775688, -8956.245252, 1000, -4000]:
        delta = isce3.core.timeDelta(dt=value)

        assert( delta.getTotalSeconds() == value)
        assert( delta.getTotalMinutes() == (value / 60.))
        assert( delta.getTotalHours() == (value / 3600.))
        assert( delta.getTotalDays() == (value / 86400.))

    return

if __name__ == '__main__':
    testConstruction()
    testTimeDelta()
    testComparison()
    testAddition()
    testTimeDelta()
    testUnits()

