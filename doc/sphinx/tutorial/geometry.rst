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


.. _invmap:

Inverse mapping example - locating corner reflectors
----------------------------------------------------

In this example, we will demonstrate the inverse mapping algorithm by using it to determine the location of a known target in a radar image.

.. code-block:: python

   from isceextension import pyDateTime, pyEllipsoid

   orbit 
