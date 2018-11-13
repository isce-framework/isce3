:orphan:

.. title:: Geometry Tutorial

Radar Geometry Tutorial
=======================

This tutorial is organized in 3 sections.

1. :ref:`usingpyorbit`
2. :ref:`forwardmap`
3. :ref:`invmap`

.. _usingpyorbit:

Working with pyOrbit
--------------------

pyOrbit is at the heart of geometric manipulation from the python level in ISCE. This is just a time-tagged collection of state vectors. In this example, we will walk through an example of constructing a pyOrbit object with a list of state vectors. The input state vectors are provided in a simple 7-column text file as shown below:

``2016-04-08T09:13:13.000000 -3752316.976337 4925051.878499 3417259.473609 3505.330104 -1842.136554 6482.122476``

``2016-04-08T09:13:23.000000 -3717067.52658 4906329.056304 3481886.455117 3544.479224 -1902.402281 6443.152265``

``........``

.. code-block:: python

   def loadOrbit(infilename):
      from isceextension import pyOrbit, pyDateTime
   
      #Create object
      orbit = pyOrbit()

      #Open file with state vectors for reading
      with open(infilename, 'r') as fid:
         linecount = 0

         #For each line in file
         for line in fid:
            vals = line.strip().split()

            #Use first time tag as reference
            if linecount == 0:
               refEpoch = pyDateTime(vals[0])
               orbit.refEpoch = refEpoch

            #Create pyDateTime. Can also be Python datetime
            tstamp = pyDateTime(vals[0])

            #Compute time difference w.r.t reference
            tstampFromRef = (tstamp-refEpoch).getTotalSeconds()
            
            #Set the state vector
            orbit.addStateVector(tstampFromRef, 
                            [float(x) for x in vals[1:4]],
                            [float(x) for x in vals[4:7]])

            #Increment line counter
            linecount += 1

      return orbit


.. _forwardmap:

Forward mapping example - determining bounding boxes
----------------------------------------------------

In this example, we will demonstrate the forward mapping algorithm by using it to determine approximate bounding boxes on the ground.

.. code-block:: python

   from isceextension import pyDateTime, pyOrbit
   "coming soon"
   "..."

.. _invmap:

Inverse mapping example - locating corner reflectors
----------------------------------------------------

In this example, we will demonstrate the inverse mapping algorithm by using it to determine the location of a known target in a radar image.

.. code-block:: python

   from isceextension import (pyOrbit, pyDateTime, 
                             pyEllipsoid, pyImageMode,
                             pyPoly2d, py_geo2rdr)
   import numpy as np

   ##Load orbit
   orbit = loadOrbit('input_orbit.txt')

   ## Targets to locate in radar image
   targets = [[131.55, 32.85, 475.],
              [131.65, 32.95, 150.]]

   #Radar wavelength
   wvl = 0.06

   #Right looking
   side = -1

   ##Fake product with relevant metadata
   mode = pyImageMode()
   mode.setDimensions([1500,1000])
   mode.prf = 1000.
   mode.rangeBandwidth = 20.0e6
   mode.wavelength = wvl
   mode.startingRange = 8.0e5
   mode.rangePixelSpacing = 10.
   mode.numberAzimuthLooks = 10
   mode.numberRangeLooks = 10
   mode.startAzTime = pyDateTime("2016-04-08T09:13:55.454821")
   mode.endAzTime = pyDateTime("2016-04-08T09:14:10.454821")

   t0 = (mode.startAzTime - orbit.refEpoch).getTotalSeconds()

   ##Create doppler polynomial - zero doppler for now
   doppler = pyPoly2d(azimuthOrder=0, rangeOrder=0,
                   azimuthMean = 0., rangeMean = 0.,
                   azimuthNorm = 1., rangeNorm = 1.)
   doppler.coeffs = [0.]

   ##Create ellipsoid - WGS84 by default
   ellps = pyEllipsoid()
   for targ in targets:
      #Convert from degrees to radians
      llh = [np.radians(targ[0]), np.radians(targ[1]), targ[2]]

      #Estimate target position
      taz, rng = py_geo2rdr(llh, ellps, orbit, doppler, mode, 
                              threshold=1.0e-8,
                              maxiter=51,
                              dR=1.0e-8)

      #Line number
      print('Target at: {0} {1} {2}'.format(*targ))
      print('Estimated line number: {0}'.format((taz - t0) * mode.prf/mode.numberAzimuthLooks))
      print('Estimated pixel number: {0}'.format((rng - mode.startingRange)/mode.rangePixelSpacing / mode.numberRangeLooks))
