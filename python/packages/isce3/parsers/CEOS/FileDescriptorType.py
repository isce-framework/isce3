###General File Descriptor
def FileDescriptorType():
    '''
    SAR Leaderfile Descriptor Record.
    http://www.ga.gov.au/__data/assets/pdf_file/0019/11719/GA10287.pdf Pg:3-23.
    '''
    from .BasicTypes import (BlankType, StringType, IntegerType, BinaryType, MultiType)  
    from .CEOSHeaderType import CEOSHeaderType

    return MultiType( CEOSHeaderType().mapping + 
                    [('ASCIIFlag' , StringType(2)),
                     ('blanks1'   , BlankType(2)),
                     ('FormatControlDocID', StringType(12)),
                     ('FormatControlDocRevision', StringType(2)),
                     ('FileDesignDescriptorRevision', StringType(2)),
                     ('LogicalVolumeSoftwareVersion', StringType(12)),
                     ('FileNumber', BinaryType('>i4')),
                     ('FileName', StringType(16)),
                     ('RecordSequenceType', StringType(4)),
                     ('SequenceNumberLocation', IntegerType(8)),
                     ('SequenceNumberLength', IntegerType(4)),
                     ('RecordCodeType', StringType(4)),
                     ('RecordCodeLocation', IntegerType(8)),
                     ('RecordCodeLength', IntegerType(4)),
                     ('RecordFieldType', StringType(4)),
                     ('RecordFieldLocation', IntegerType(8)),
                     ('RecordFieldLength', IntegerType(4)),
                     ('blanks2', BlankType(68))])
