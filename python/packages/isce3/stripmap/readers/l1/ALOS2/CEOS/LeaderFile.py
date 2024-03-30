class LeaderFile(object):
    '''
    Class for parsing ALOS-2 L1.1 CEOS Leaderfile.
    '''

    def __init__(self, filename):
        '''
        Initialize object with leader filename.
        '''
        import os

        # Save file name
        self.name = filename

        # Save file size in bytes
        self.size = os.stat(filename).st_size

        # Leader file always seems to consist of same set of records.
        # https://www.eorc.jaxa.jp/ALOS-2/en/doc/fdata/PALSAR-2_xx_Format_CEOS_E_f.pdf

        with open(self.name, 'rb') as fid:
            # Leader file descriptor
            self.description = self.parseFileDescriptor(fid)

            # Dataset summary
            self.summary = self.parseDatasetSummary(fid)

            # Platform position
            self.platformPosition = self.parsePlatformPosition(fid)

            # Attitude data
            self.attitude = self.parseAttitude(fid)

            # Calibration data
            self.calibration = self.parseCalibration(fid)

    def parseFileDescriptor(self, fid):
        '''
        Parse SAR Leaderfile descriptor record.
        '''
        from isce3.parsers.CEOS.LeaderFileDescriptorType import \
            LeaderFileDescriptorType

        # Description of record - seems to be common across missions
        record = LeaderFileDescriptorType()

        # Read the record
        record.fromfile(fid)

        # Sensor specific validators
        # https://www.eorc.jaxa.jp/ALOS-2/en/doc/fdata/PALSAR-2_xx_Format_CEOS_E_f.pdf
        assert (record.RecordSequenceNumber == 1)
        assert (record.FirstRecordType == 11)
        assert (record.RecordTypeCode == 192)
        assert (record.SecondRecordSubType == 18)
        assert (record.ThirdRecordSubType == 18)
        assert (record.RecordLength == 720)
        assert (record.RecordSequenceType == 'FSEQ')
        assert (record.RecordCodeType == 'FTYP')
        assert (record.RecordFieldType == 'FLGT')

        # Check length of record
        assert (fid.tell() == record.RecordLength)

        # Return the validated record
        return record

    def parseDatasetSummary(self, fid):
        '''
        Parse Dataset Summary record.
        '''
        from .DataSetSummaryRecordType import DatasetSummaryRecordType

        # First ensure that spec says there is only one record
        assert (self.description.NumberDatasetSummaryRecords == 1)

        # Description of record - customized record from local directory
        record = DatasetSummaryRecordType()

        # Read the record
        record.fromfile(fid)

        # Sensor specific validators
        # https://www.eorc.jaxa.jp/ALOS-2/en/doc/fdata/PALSAR-2_xx_Format_CEOS_E_f.pdf
        assert (record.RecordSequenceNumber == 2)
        assert (record.FirstRecordType == 18)
        assert (record.RecordTypeCode == 10)
        assert (record.SecondRecordSubType == 18)
        assert (record.ThirdRecordSubType == 20)
        assert (record.RecordLength == 4096)
        assert (record.SensorPlatformMissionIdentifier == 'ALOS2')
        # Specific check for ALOS-2 stripmap, normal observation mode
        assert (record.SensorIDAndMode[0:9] == 'ALOS2 -L ')
        assert (record.RangePulseCodeSpecifier == "LINEAR FM CHIRP")
        assert (record.QuantizationDescriptor == "UNIFORM I,Q")
        assert (record.ProductTypeSpecifier == "BASIC IMAGE")
        assert (record.DataInputSource == "ONLINE")
        assert (record.LineContentIndicator == "RANGE")
        assert (0 <= record.AntennaBeamNumber <= 22)

        # Check against byte offset
        assert (fid.tell() == sum([
            self.description.RecordLength,
            self.description.DatasetSummaryRecordLength]))

        # Return the validated record
        return record

    def parsePlatformPosition(self, fid):
        '''
        Parse the platform position record.
        '''
        from .PlatformPositionDataRecordType import (
            PlatformPositionDataRecordHeaderType,
            PlatformPositionDataRecordStateVectorType,
            PlatformPositionDataRecordTrailerType)

        # Simple container
        class Container(object):
            pass

        record = Container()

        # First ensure that spec saus there is only one record
        assert (self.description.NumberPlatformPositionRecords == 1)

        # Start with the header of the record
        header = PlatformPositionDataRecordHeaderType()

        # Read the header
        header.fromfile(fid)

        # ALOS specific validators
        # https://www.eorc.jaxa.jp/ALOS-2/en/doc/fdata/PALSAR-2_xx_Format_CEOS_E_f.pdf
        assert (header.RecordSequenceNumber == 3)
        assert (header.FirstRecordType == 18)
        assert (header.RecordTypeCode == 30)
        assert (header.SecondRecordSubType == 18)
        assert (header.ThirdRecordSubType == 20)
        assert (header.RecordLength == 4680)
        assert (header.OrbitalElementsDesignator in ['0', '1', '2'])
        assert (header.NumberOfDataPoints == 28)
        assert (header.TimeIntervalBetweenDataPointsInSec == 60.0)
        record.header = header

        # Now parse the state vectors individually
        svs = []
        for ii in range(header.NumberOfDataPoints):
            sv = PlatformPositionDataRecordStateVectorType()
            sv.fromfile(fid)
            svs.append(sv)

        record.statevectors = svs

        # ALOS specific custom trailer
        trailer = PlatformPositionDataRecordTrailerType()
        trailer.fromfile(fid)
        assert (trailer.LeapSecondFlag in [0, 1])

        # Check against byte offset
        assert (fid.tell() == sum(
            [self.description.RecordLength,
             self.description.DatasetSummaryRecordLength,
             self.description.PlatformPositionRecordLength]))

        return record

    def parseAttitude(self, fid):
        from .AttitudeDataRecordType import (AttitudeDataRecordHeaderType,
                                             AttitudeDataRecordStateVectorType,
                                             AttitudeDataRecordTrailerType)

        # Simple container
        class Container(object):
            pass

        record = Container()

        # First ensure that spec saus there is only one record
        assert (self.description.NumberAttitudeRecords == 1)

        # Start with the header of the record
        header = AttitudeDataRecordHeaderType()

        # Read the header
        header.fromfile(fid)

        # ALOS specific validators
        # https://www.eorc.jaxa.jp/ALOS-2/en/doc/fdata/PALSAR-2_xx_Format_CEOS_E_f.pdf
        assert (header.RecordSequenceNumber == 4)
        assert (header.FirstRecordType == 18)
        assert (header.RecordTypeCode == 40)
        assert (header.SecondRecordSubType == 18)
        assert (header.ThirdRecordSubType == 20)
        assert (header.NumberOfAttitudeDataPoints in [22,62])
        record.header = header

        # Now parse the state vectors individually
        svs = []
        for ii in range(header.NumberOfAttitudeDataPoints):
            sv = AttitudeDataRecordStateVectorType()
            sv.fromfile(fid)
            svs.append(sv)

        record.statevectors = svs

        # ALOS specific custom trailer
        trailerLength = \
            header.RecordLength - 16 - 120 * header.NumberOfAttitudeDataPoints 
        trailer = AttitudeDataRecordTrailerType(trailerLength)
        trailer.fromfile(fid)

        # Check against byte offset
        assert (fid.tell() == sum(
            [self.description.RecordLength,
             self.description.DatasetSummaryRecordLength,
             self.description.PlatformPositionRecordLength,
             self.description.AttitudeRecordLength]))

        return record

    def parseCalibration(self, fid):
        '''
        Parse Calibration Data Record.
        '''
        from .RadiometricRecordType import RadiometricRecordType

        # Simple container
        class Container(object):
            pass

        record = Container()


        # Start with the header of the record
        header = RadiometricRecordType()

        # Read the header
        header.fromfile(fid)

        record.header = header

        return record

    def parseFacilityData(self, fid):
        '''
        This is very mission specfic. 
        Did not find public document with a detailed description of this.
        Last record in leader and not needed for processing. So skipping transcoding it ...
        '''
        raise NotImplementedError()
