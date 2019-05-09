#!/usr/bin/env python3

def testConstruction():
    import datetime
    from isce3.extensions.isceextension import pyDateTime

    #Reference datetime
    refobjs = [datetime.datetime(2017,5,12,1,12,30,141592),
               datetime.datetime(2011,12,31,0,23,59,999999),
               datetime.datetime(2019,1,1,0,0,0,1)]
               
   
    for refobj in refobjs:
        #Create using string
        pyobj = pyDateTime(refobj.isoformat())

        #Create datetime from pyDateTime
        reconsobj = datetime.datetime.strptime(pyobj.isoformat()[:-3], "%Y-%m-%dT%H:%M:%S.%f")
    
        assert abs((reconsobj - refobj).total_seconds()) < 1.0e-6

        #Create using datetime
        pyobj = pyDateTime(refobj)

        #Create datetime from pyDateTime
        reconsobj = datetime.datetime.strptime(pyobj.isoformat()[:-3], "%Y-%m-%dT%H:%M:%S.%f")

        assert abs((reconsobj - refobj).total_seconds()) < 1.0e-6


def testComparison():
    from isce3.extensions.isceextension import pyDateTime

    dtime1 = pyDateTime("2017-05-12T01:12:30.141592")
    dtime2 = pyDateTime("2017-05-13T02:12:33.241592")
   
    ###Copy from dtime1
    dtime3 = pyDateTime()
    dtime3.set(dtime1.isoformat())

    assert (dtime1 < dtime2)
    assert (dtime2 > dtime1)
    assert (dtime1 >= dtime3)
    assert (dtime1 <= dtime3)
    assert (dtime1 == dtime3)
    assert (dtime1 != dtime2)


def testTimeDelta():
    from isce3.extensions.isceextension import pyDateTime, pyTimeDelta

    dtime1 = pyDateTime("2017-05-12T01:12:30.141592")
    dtime2 = pyDateTime("2017-05-13T02:12:33.241592")
    delta = dtime2 - dtime1

    assert abs(delta.getTotalSeconds() - 90003.1) < 1.0e-8

def testAddition():
    from isce3.extensions.isceextension import pyDateTime, pyTimeDelta

    dtime1 = pyDateTime("2017-05-12T01:12:30.141592")
    delta = pyTimeDelta(3.0)

    dtime2 = dtime1 + delta
    assert(dtime2.isoformat() == "2017-05-12T01:12:33.141592000")

    delta = pyTimeDelta(-4.0)
    dtime2 = dtime1 + delta
    assert(dtime2.isoformat() == "2017-05-12T01:12:26.141592000")

def testUnits():
    from isce3.extensions.isceextension import pyTimeDelta

    for value in [90456.767775688, -8956.245252, 1000, -4000]:
        delta = pyTimeDelta(value)

        assert( delta.getTotalSeconds() == value)
        assert( delta.getTotalMinutes() == (value / 60.))
        assert( delta.getTotalHours() == (value / 3600.))
        assert( delta.getTotalDays() == (value / 86400.))

if __name__ == '__main__':
    testConstruction()
    testTimeDelta()
    testComparison()
    testAddition()
