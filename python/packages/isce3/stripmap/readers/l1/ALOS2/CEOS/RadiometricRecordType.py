def RadiometricRecordType():
    '''
    Radiometric Record.
    https://www.eorc.jaxa.jp/ALOS-2/en/doc/fdata/PALSAR-2_xx_Format_CEOS_E_f.pdf
    '''
    from isce3.parsers.CEOS.BasicTypes import (
        BlankType, BinaryType, StringType, IntegerType, FloatType, MultiType)
    from isce3.parsers.CEOS.CEOSHeaderType import CEOSHeaderType

    #Common part of CEOS header (modified)
    inlist = CEOSHeaderType().mapping 
    inlist += [('SARChannelIndicator', BinaryType('>i4')),
               ('NumberOfDataSets', BinaryType('>i4')),
               ('CalibrationFactor', FloatType(16)),
               ('RealPartOfDT1,1', FloatType(16)),
               ('ImaginaryPartOfDT1,1', FloatType(16)),
               ('RealPartOfDT1,2', FloatType(16)),
               ('ImaginaryPartOfDT1,2', FloatType(16)),
               ('RealPartOfDT2,1', FloatType(16)),
               ('ImaginaryPartOfDT2,1', FloatType(16)),
               ('RealPartOfDT2,2', FloatType(16)),
               ('ImaginaryPartOfDT2,2', FloatType(16)),
               ('RealPartOfDR1,1', FloatType(16)),
               ('ImaginaryPartOfDR1,1', FloatType(16)),
               ('RealPartOfDR1,2', FloatType(16)),
               ('ImaginaryPartOfDR1,2', FloatType(16)),
               ('RealPartOfDR2,1', FloatType(16)),
               ('ImaginaryPartOfDR2,1', FloatType(16)),
               ('RealPartOfDR2,2', FloatType(16)),
               ('ImaginaryPartOfDR2,2', FloatType(16)),
               ('SkipBytes', BlankType(9577))]

    return MultiType( inlist )

