##CEOS Header Type
def CEOSHeaderType():
    '''
    This is common to all CEOS records.
    Reference: http://www.ga.gov.au/__data/assets/pdf_file/0019/11719/GA10287.pdf.
    '''
    from .BasicTypes import BinaryType, MultiType
    return MultiType([('RecordSequenceNumber', BinaryType('>i4')),
                      ('FirstRecordType', BinaryType('>B')),
                      ('RecordTypeCode',  BinaryType('>B')),
                      ('SecondRecordSubType', BinaryType('>B')),
                      ('ThirdRecordSubType', BinaryType('>B')),
                      ('RecordLength', BinaryType('>i4'))])


