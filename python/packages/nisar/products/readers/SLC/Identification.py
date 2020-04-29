from ..Base.Identification import Identification as BaseId

class Identification(BaseId):
    '''
    Simple object to hold identification information for NISAR SLC products.
    '''
    def __init__(self, *args, **kw):
        '''
        Identify given object as relevant NISAR product.
        '''
        self.absoluteOrbitNumber = None
        self.trackNumber = None
        self.frameNumber = None
        self.zdStartTime = None
        self.zdStopTime = None
        self.boundingPolygon = None
        super().__init__(*args, **kw)


    def unpack(self, h5grp):
        '''
        Populate self with hdf5 group.
        '''
        from nisar.h5 import extractScalar, bytestring, extractWithIterator
        from isce3.extensions.isceextension import pyDateTime

        BaseId.unpack(self, h5grp)

        self.absoluteOrbitNumber = extractScalar(h5grp, 'absoluteOrbitNumber',
                                      int, self.context['info'],
                                      'Absolute orbit number could not be identified')
        self.trackNumber = extractScalar(h5grp, 'trackNumber',
                                      int, self.context['info'],
                                      'Track number could not be identified')
        self.frameNumber = extractScalar(h5grp, 'frameNumber',
                                      int, self.context['info'],
                                      'Frame number could not be identified')
        self.zdStartTime = extractScalar(h5grp, 'zeroDopplerStartTime',
                                      bytestring, self.context['error'])
        self.zdStartTime = pyDateTime(self.zdStartTime)

        self.zdEndTime = extractScalar(h5grp, 'zeroDopplerEndTime',
                                      bytestring, self.context['error'])
        self.zdEndTime = pyDateTime(self.zdEndTime)

        self.boundingPolygon = extractScalar(h5grp, 'boundingPolygon',
                                      bytestring, self.context['info'],
                                      'No bounding polygon could be identified')
