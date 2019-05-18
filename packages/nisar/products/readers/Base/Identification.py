# -*- coding: utf-8 -*-

import journal

class Identification(object):
    '''
    Simple object to hold identification information for NISAR products.
    '''
    def __init__(self, inobj, path='.', context=None):
        '''
        Identify given object as relevant NISAR product.
        '''

        self.missionId = None
        self.productType = None
        self.absoluteOrbitNumber = None
        self.trackNumber = None
        self.frameNumber = None
        self.lookDirection = None
        self.orbitPassDirection = None
        self.zdStartTime = None
        self.zdStopTime = None
        self.boundingPolygon = None
        self.listOfFrequencies = None

        ###Information from mission planning
        self.isUrgentObservation = None
        self.plannedDataTake = None
        self.plannedObservation = None
        
        
        
        import h5py

        #Any logging context 
        if context is None:
            context = { 'info': journal.info('nisar.reader'),
                        'debug': journal.debug('nisar.reader'),
                        'error': journal.error('nisar.reader')}

        self.context = context

        #User has an open HDF5 file and is looking into it
        if isinstance(inobj, h5py.Group):
            self.unpack(inobj[path])
        #User provides HDF5 file and path inside it 
        elif isinstance(inobj, str):
            with h5py.File(inobj, 'r') as fid:
                self.unpack(fid, path)
        

    def unpack(self, h5grp):
        '''
        Populate self with hdf5 group.
        '''
        from nisar.h5 import extractScalar, bytestring, extractWithIterator
        from isce3.extensions.isceextension import pyDateTime

        self.missionId = extractScalar(h5grp, 'missionId', 
                                      bytestring, self.context['info'],
                                      'Mission could not be identified')
        self.productType = extractScalar(h5grp, 'productType', 
                                      bytestring, self.context['error'],
                                      'Product type could not be determined')
        self.absoluteOrbitNumber = extractScalar(h5grp, 'absoluteOrbitNumber',
                                      int, self.context['info'],
                                      'Absolute orbit number could not be identified')
        self.trackNumber = extractScalar(h5grp, 'trackNumber',
                                      int, self.context['info'],
                                      'Track number could not be identified')
        self.frameNumber = extractScalar(h5grp, 'frameNumber',
                                      int, self.context['info'],
                                      'Frame number could not be identified')
        self.lookDirection = extractScalar(h5grp, 'lookDirection',
                                      bytestring, self.context['error'],
                                      'Look direction could not be identified')
        self.orbitPassDirection = extractScalar(h5grp, 'orbitPassDirection',
                                      bytestring, self.context['info'],
                                      'Pass direction could not be identified')
        self.zdStartTime = extractScalar(h5grp, 'zeroDopplerStartTime',
                                      bytestring, self.context['error'])
        self.zdStartTime = pyDateTime(self.zdStartTime)

        self.zdEndTime = extractScalar(h5grp, 'zeroDopplerEndTime',
                                      bytestring, self.context['error'])
        self.zdEndTime = pyDateTime(self.zdEndTime)

        self.boundingPolygon = extractScalar(h5grp, 'boundingPolygon',
                                      bytestring, self.context['info'],
                                      'No bounding polygon could be identified')

        self.listOfFrequencies = extractWithIterator(h5grp, 'listOfFrequencies',
                                      bytestring, self.context['error'],
                                      'List of frequencies could not be determined')

        ###Mission planning info to be added
        ###Processing type info to be added

                

# end of file
