#!/usr/bin/env python3

def testConstruction():
    import datetime
    from isceextension import pyDateTime

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


def testTimeDelta():
    from isceextension import pyDateTime, pyTimeDelta

    dtime1 = pyDateTime("2017-05-12T01:12:30.141592")
    dtime2 = pyDateTime("2017-05-13T02:12:33.241592")
    delta = dtime2 - dtime1

    assert abs(delta.getTotalSeconds() - 90003.1) < 1.0e-8

if __name__ == '__main__':
    testConstruction()
    testTimeDelta()
