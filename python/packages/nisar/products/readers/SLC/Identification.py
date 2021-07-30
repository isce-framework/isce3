from ..Base.Identification import Identification as BaseId

class Identification(BaseId):
    '''
    Simple object to hold identification information for NISAR SLC products.
    '''
    def __init__(self, *args, **kw):
        '''
        Identify given object as relevant NISAR product.
        '''
        self.trackNumber = None
        self.frameNumber = None
        super().__init__(*args, **kw)


    def unpack(self, h5grp):
        '''
        Populate self with hdf5 group.
        '''
        from nisar.h5 import extractScalar, bytestring

        BaseId.unpack(self, h5grp)

        self.trackNumber = extractScalar(h5grp, 'trackNumber',
                                      int, self.context['info'],
                                      'Track number could not be identified')
        self.frameNumber = extractScalar(h5grp, 'frameNumber',
                                      int, self.context['info'],
                                      'Frame number could not be identified')
