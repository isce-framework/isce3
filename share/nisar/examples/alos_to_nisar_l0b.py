#!/usr/bin/env python3
import argparse
import datetime
import glob
import h5py
import isce3
import numpy
import os
from warnings import warn
from isce3.stripmap.readers.l0raw.ALOS.CEOS import ImageFile, LeaderFile
from nisar.antenna import CalPath
from nisar.products import descriptions
from nisar.products.readers.Raw import Raw
from nisar.products.readers.Raw.Raw import get_rcs2body
from nisar.workflows.focus import make_doppler_lut
from isce3.core import speed_of_light

def cmdLineParse():
    '''
    Command line parser.
    '''
    parser = argparse.ArgumentParser(description="Package ALOS L0 stripmap data into NISAR L0B HDF5",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--indir', dest='indir', type=str,
                        help="Folder containing one ALOS L0 module",
                        required=True)
    parser.add_argument('-o', '--outh5', dest='outh5', type=str,
                        help="Name of output file. If not provided, will be determined from ALOS granule",
                        default=None)
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help="Use more rigorous parser to check magic bytes aggresively",
                        default=False)
    parser.add_argument("--simulate-gap", dest="gap_width_usec", type=float,
                        help="Simulate a transmit gap of the given width in microseconds",
                        default=0.0)
    parser.add_argument("--gap-location", type=float, default=0.5,
                        help="Location of simulated gap given as a fraction of the swath.")

    inps = parser.parse_args()
    if not os.path.isdir(inps.indir):
        raise ValueError('{0} does not appear to be a directory'.format(inps.indir))

    if inps.outh5 is None:
        print('HDF5 output granule name will be determined on fly and created in cwd')

    return inps

def getALOSFilenames(indir):
    '''
    Parse the contents of a given directory to separate out leader and image files.
    '''
    filenames = {}

    ##First look for the leader file
    flist = glob.glob(os.path.join(indir, 'LED-ALPSRP*1.0__*'))
    if len(flist) == 0:
        raise ValueError('No leader files found in folder {0}'.format(indir))
    elif len(flist) > 1:
        raise ValueError('Multiple leader files in folder {0}'.format(indir))

    filenames['leaderfile']  = flist[0]
    pattern = os.path.basename(flist[0])[4:]


    ##Look for polarizations
    for pol in ['HH', 'HV', 'VV', 'VH']:
        flist = glob.glob(os.path.join(indir, 'IMG-{0}-{1}'.format(pol, pattern)))
        if len(flist) == 1:
            print('Found Polarization: {0}'.format(pol))
            filenames[pol] = flist[0]


    #If no image files were found
    if len(filenames) == 1:
        raise ValueError('No image files were found in folder: {0}'.format(indir))

    filenames['defaulth5'] = '{0}.h5'.format(pattern)

    return filenames


def parseLeaderFile(filenames):
    '''
    Parse leader file and check values against polarizations.
    '''

    try:
        ldr = LeaderFile.LeaderFile(filenames['leaderfile'])
    except AssertionError as msg:
        print(msg)
        raise AssertionError('Error parsing ALOS raw leader file: {0}'.format(filenames['leaderfile']))


    ##Checks to ensure that the number of polarizations is consistent
    numpol = len(filenames) - 2 #Subtract leader and defaulth5 name
    if numpol != ldr.summary.NumberOfSARChannels:
        raise ValueError('Number of image files discovered is inconsistent with Leader File')

    return ldr


def alos_quaternion(t, euler_ypr_deg, orbit):
    """Convert ALOS attitude (body to TCN(ECI)) to quaternion (body to XYZ(ECF))

    t               Time (datetime.datetime)
    euler_ypr_deg   (yaw, pitch, roll) Euler sequence (in degrees)
    orbit           Orbit ephemeris (isce3.core.Orbit)
    """
    ti = (t - orbit.reference_epoch).total_seconds()
    pos, vel = orbit.interpolate(ti)

    # ALOS format doc doesn't really describe coordinate frames.
    # Assuming body to TCN, given in inertial frame.
    vel_eci = isce3.core.velocity_eci(pos, vel)

    # Based on the following reference I assume local vertical is geocentric
    # (does not depend on latitude or Ellipsoid).
    # https://issfd.org/ISSFD_2011/S12-Orbit.Dynamics.3-ODY3/S12_P5_ISSFD22_PF_068.pdf
    tcn2xyz = isce3.core.Basis(pos, vel_eci).asarray()

    yaw, pitch, roll = numpy.radians(euler_ypr_deg)
    body2tcn = isce3.core.EulerAngles(yaw, pitch, roll).to_rotation_matrix()
    body2xyz = tcn2xyz.dot(body2tcn)
    return isce3.core.Quaternion(body2xyz)


def get_alos_orbit(ldr: LeaderFile.LeaderFile) -> isce3.core.Orbit:
    hdr = ldr.platformPosition.header
    tref = datetime.datetime(hdr.YearOfDataPoint, hdr.MonthOfDataPoint,
                             hdr.DayOfDataPoint)
    t0, dt = hdr.SecondsOfDay, hdr.TimeIntervalBetweenDataPointsInSec
    times, svs = [], []
    for i, sv in enumerate(ldr.platformPosition.statevectors):
        t = t0 + i * dt * 1.0
        times.append(t)
        timestamp = isce3.core.DateTime(tref) + isce3.core.TimeDelta(seconds=t)
        svs.append(isce3.core.StateVector(
            datetime = timestamp,
            position = [sv.PositionXInm, sv.PositionYInm, sv.PositionZInm],
            velocity = [sv.VelocityXInmpers, sv.VelocityYInmpers, sv.VelocityZInmpers]
        ))
    # Use tref as epoch, not time of first sample.
    return isce3.core.Orbit(svs, isce3.core.DateTime(tref), type='DOE')


def set_h5_orbit(group: h5py.Group, orbit: isce3.core.Orbit):
    orbit.save_to_h5(group)
    # acceleration not used/contained in Orbit object
    dset = group.create_dataset("acceleration",
                                data=numpy.zeros_like(orbit.velocity))
    dset.attrs["units"] = numpy.string_("meters per second squared")
    dset.attrs["description"] = numpy.string_("GPS state vector acceleration")


def getset_attitude(group: h5py.Group, ldr: LeaderFile.LeaderFile,
                    orbit: isce3.core.Orbit):
    """Read attitude from LeaderFile and write to NISAR HDF5 group.
    Assumes orbit.reference_epoch is in same year as attitude data.
    """
    assert len(ldr.attitude.statevectors) > 0
    # ASF has no ALOS data crossing New Years in its archive, but check anyway
    # that day of year doesn't roll over.
    days = [sv.DayOfYear for sv in ldr.attitude.statevectors]
    assert all(numpy.diff(days) >= 0), "Unexpected roll over in day of year."

    # build quaternion for antenna to spacecraft body (RCS to Body) per
    # mechanical boresight angle (MB).  Not sure if "NominalOffNadirAngle" is
    # defined in exactly the same way, but it must be close for ALOS since
    # it's steered to zero-Doppler.
    q_rcs2body = get_rcs2body(el_deg=ldr.summary.NominalOffNadirAngle,
                              side='right')
    print(f"Using off-nadir angle {ldr.summary.NominalOffNadirAngle} degrees"
          f" for beam number {ldr.summary.AntennaBeamNumber}.")

    times = []  # time stamps, seconds rel orbit epoch
    rpys = []   # (roll, pitch, yaw) Euler angle tuples, degrees
    qs = []     # (q0,q1,q2,q3) quaternion arrays
    for sv in ldr.attitude.statevectors:
        dt = isce3.core.TimeDelta(datetime.timedelta(
            days = sv.DayOfYear - 1,
            seconds = 0.001 * sv.MillisecondsOfDay))
        t = isce3.core.DateTime(orbit.reference_epoch.year, 1, 1) + dt
        rpy = (sv.RollInDegrees, sv.PitchInDegrees, sv.YawInDegrees)
        rpys.append(rpy)
        # Use time reference as orbit.
        times.append((t - orbit.reference_epoch).total_seconds())
        q = alos_quaternion(t, rpy[::-1], orbit) * q_rcs2body
        qs.append([q.w, q.x, q.y, q.z])

    # Write to HDF5.  isce3.core.Quaternion.save_to_h5 doesn't really cut it,
    # so don't bother constructing it.
    ds = group.create_dataset("angularVelocity", data=numpy.zeros((len(qs), 3)))
    ds.attrs["description"] = numpy.string_(
        "Attitude angular velocity vectors (wx, wy, wz)")
    ds.attrs["units"] = numpy.string_("radians per second")

    ds = group.create_dataset("attitudeType", data=numpy.string_("Custom"))
    ds.attrs["description"] = numpy.string_(
        'Attitude type, either "FRP", "NRP", "PRP, or "Custom", where "FRP"'
        ' stands for Forecast Radar Pointing, "NRP" is Near Real-time'
        ' Pointing, and "PRP" is Precise Radar Pointing')

    ds = group.create_dataset("eulerAngles", data=numpy.array(rpys))
    ds.attrs["description"] = numpy.string_(
        "Attitude Euler angles (roll, pitch, yaw)")
    ds.attrs["units"] = numpy.string_("degrees")

    ds = group.create_dataset("quaternions", data=numpy.array(qs))
    ds.attrs["description"] = numpy.string_("Attitude quaternions (q0, q1, q2, q3)")
    ds.attrs["units"] = numpy.string_("unitless")

    ds = group.create_dataset("time", data=numpy.array(times))
    ds.attrs["description"] = numpy.string_(
        "Time vector record. This record contains the time corresponding to"
        " attitude and quaternion records")
    ds.attrs["units"] = numpy.string_(
        f"seconds since {orbit.reference_epoch.isoformat()}")


ident_descriptions = {
  'absoluteOrbitNumber': 'Absolute orbit number',
  'boundingPolygon': descriptions.bounding_polygon,
  'diagnosticModeFlag': 'Indicates if the radar operation mode is a diagnostic '
                        'mode (1-2) or DBFed science (0): 0, 1, or 2',
  'granuleId': 'Unique granule identification name',
  'instrumentName': 'Name of the instrument used to collect the remote sensing '
                    'data provided in this product',
  'isDithered': '"True" if the pulse timing was varied (dithered) during '
                'acquisition, "False" otherwise.',
  'isGeocoded': 'Flag to indicate if the product data is in the radar geometry '
                '("False") or in the map geometry ("True")',
  'isMixedMode': '"True" if this product is a composite of data collected in '
                 'multiple radar modes, "False" otherwise.',
  'isUrgentObservation': 'Flag indicating if observation is nominal ("False") '
                         'or urgent ("True")',
  'listOfFrequencies': 'List of frequency layers available in the product',
  'lookDirection': 'Look direction, either "Left" or "Right"',
  'missionId': 'Mission identifier',
  'orbitPassDirection': 'Orbit direction, either "Ascending" or "Descending"',
  'plannedDatatakeId': 'List of planned datatakes included in the product',
  'plannedObservationId': 'List of planned observations included in the '
                          'product',
  'processingCenter': 'Data processing center',
  'processingDateTime': 'Processing UTC date and time in the format '
                        'YYYY-MM-DDTHH:MM:SS',
  'processingType': 'NOMINAL (or) URGENT (or) CUSTOM (or) UNDEFINED',
  'productLevel': 'Product level. L0A: Unprocessed instrument data; L0B: '
                  'Reformatted, unprocessed instrument data; L1: Processed '
                  'instrument data in radar coordinates system; and L2: '
                  'Processed instrument data in geocoded coordinates system',
  'productSpecificationVersion': 'Product specification version which '
                                 'represents the schema of this product',
  'productType': 'Product type',
  'productVersion': 'Product version which represents the structure of the '
                    'product and the science content governed by the '
                    'algorithm, input data, and processing parameters',
  'radarBand': 'Acquired frequency band',
  'zeroDopplerEndTime': 'Azimuth stop time of the product',
  'zeroDopplerStartTime': 'Azimuth start time of the product'
}


def populateIdentification(ident: h5py.Group, ldr: LeaderFile.LeaderFile):
    """Populate L0B identification data.  Fields in
    {"boundingPolygon"," zeroDopplerStartTime", "zeroDopplerEndTime"}
    are populated with dummy values.
    """
    # scalar
    ident.create_dataset('diagnosticModeFlag', data=numpy.uint8(0))
    ident.create_dataset('isGeocoded', data=numpy.string_("False"))
    ident.create_dataset('listOfFrequencies', data=numpy.string_(["A"]))
    ident.create_dataset('lookDirection', data = numpy.string_("right"))
    ident.create_dataset('missionId', data=numpy.string_("ALOS"))
    direction = "ascending" if ldr.summary.TimeDirectionIndicatorAlongLine[0] == "A" else "descending"
    ident.create_dataset('orbitPassDirection', data=numpy.string_(direction))
    ident.create_dataset('processingType', data=numpy.string_("repackaging"))
    ident.create_dataset('productType', data=numpy.string_("RRSD"))
    ident.create_dataset('productVersion', data=numpy.string_("0.1.0"))
    ident.create_dataset('absoluteOrbitNumber', data=numpy.array(0, dtype='u4'))
    ident.create_dataset("isUrgentObservation", data=numpy.string_("False"))
    # shape = numberOfObservations
    ident.create_dataset("plannedObservationId", data=numpy.string_(["0"]))
    # shape = numberOfDatatakes
    ident.create_dataset("plannedDatatakeId", data=numpy.string_(["0"]))
    # Will override these three later.
    ident.create_dataset("boundingPolygon", data=numpy.string_("POLYGON EMPTY"))
    ident.create_dataset("zeroDopplerStartTime", data=numpy.string_(
        "2007-01-01 00:00:00.0000000"))
    ident.create_dataset("zeroDopplerEndTime", data=numpy.string_(
        "2007-01-01 00:00:01.0000000"))
    # fields added to spec in 2023
    ident.create_dataset("granuleId", data=numpy.string_("None"))
    ident.create_dataset("instrumentName", data=numpy.string_("PALSAR"))
    ident.create_dataset("isDithered", data=numpy.string_("False"))
    ident.create_dataset("isMixedMode", data=numpy.string_("False"))
    ident.create_dataset("processingCenter", data=numpy.string_("JPL"))
    ident.create_dataset("processingDateTime",
        data=numpy.string_(datetime.datetime.now().isoformat()[:19]))
    ident.create_dataset("productLevel", data=numpy.string_("L0B"))
    ident.create_dataset("productSpecificationVersion",
        data=numpy.string_("0.9.0"))
    ident.create_dataset("radarBand", data=numpy.string_("L"))
    
    for name, desc in ident_descriptions.items():
        ident[name].attrs["description"] = numpy.string_(desc)


def constructNISARHDF5(args, ldr):
    '''
    Build skeleton of HDF5 file using leader file information.
    '''
    with h5py.File(args.outh5, 'w-') as fid:
        lsar = fid.create_group('/science/LSAR')
        ##Fill up Identification
        ident = lsar.create_group('identification')
        populateIdentification(ident, ldr)

        ##Start populating metadata parts
        rrsd = lsar.create_group('RRSD')
        inps = rrsd.create_group('metadata/processingInformation/inputs')
        inps.create_dataset('l0aGranules', data=numpy.string_([os.path.basename(args.indir)]))

        #Start populating telemetry
        orbit_group = rrsd.create_group('lowRateTelemetry/orbit')
        orbit = get_alos_orbit(ldr)
        set_h5_orbit(orbit_group, orbit)
        attitude_group = rrsd.create_group("lowRateTelemetry/attitude")
        getset_attitude(attitude_group, ldr, orbit)


def makeDummyCalType(n, first_bcal=0, first_lcal=500, interval=1000):
    '''Create a numpy array suitable for populating calType field.
    '''
    dtype = h5py.enum_dtype(dict(CalPath.__members__), basetype="uint8")
    x = numpy.zeros(n, dtype=dtype)
    x[:] = CalPath.HPA
    x[first_bcal::interval] = CalPath.BYPASS
    x[first_lcal::interval] = CalPath.LNA
    return x


def getNominalSpacing(prf, dr, orbit, look_angle, i=0):
    """Return ground spacing along- and across-track
    """
    pos, vel = orbit.position[i], orbit.velocity[i]
    vs = numpy.linalg.norm(vel)
    ell = isce3.core.Ellipsoid()
    lon, lat, h = ell.xyz_to_lon_lat(pos)
    hdg = isce3.geometry.heading(lon, lat, vel)
    a = ell.r_dir(hdg, lat)
    ds = vs / prf * a / (a + h)
    dg = dr / numpy.sin(look_angle)
    return ds, dg


def addImagery(h5file, ldr, imgfile, pol, gap_width_usec=0.0, gap_location=0.5):
    '''
    Populate swaths segment of HDF5 file.
    ''' 

    fid = h5py.File(h5file, 'r+')
    assert(len(pol) == 2)
    txP = pol[0]
    rxP = pol[1]


    #Parse imagefile descriptor and first record.
    image = ImageFile.ImageFile(imgfile)
    firstrec = image.readNextLine()


    #Range related parameters
    fsamp = ldr.summary.SamplingRateInMHz * 1.0e6
    r0 = firstrec.SlantRangeToFirstSampleInm
    dwp_delay = 2 * r0 / speed_of_light
    print('RX data window position of the first record -> '
          f'{dwp_delay * 1e3:.4f} (msec)')
    dr = speed_of_light / (2 * fsamp)
    nPixels = image.description.NumberOfBytesOfSARDataPerRecord // image.description.NumberOfSamplesPerDataGroup
    nLines = image.description.NumberOfSARDataRecords

    sub_swaths = [[0, nPixels]]
    if gap_width_usec > 0.0:
        print(f"Chirp length is {firstrec.ChirpLengthInns * 1e3} usec")
        print(f"Simulating {gap_width_usec} usec gap")
        ngap = round(gap_width_usec * 1e-6 * fsamp)
        icenter = round(gap_location * nPixels)
        istop1 = icenter - ngap // 2
        istart2 = istop1 + ngap
        print(f"Gap located between pixels [{istop1}, {istart2})")
        # Need at least one valid sample on each side of gap.
        if (istop1 < 1) or (istart2 >= nPixels):
            raise ValueError(f"Gap parameters do not leave two valid subswaths")
        sub_swaths = [[0, istop1], [istart2, nPixels]]

    # Figure out nominal ground spacing
    prf = firstrec.PRFInmHz / 1000./ (1 + (ldr.summary.NumberOfSARChannels == 4))
    look_angle = numpy.radians(ldr.summary.NominalOffNadirAngle)
    ds, dg = getNominalSpacing(prf, dr, get_alos_orbit(ldr), look_angle)
    print('ds, dg =', ds, dg)

    freqA = '/science/LSAR/RRSD/swaths/frequencyA'
    #If this is first pol being written, add common information as well
    if freqA not in fid:
        freqA = fid.create_group(freqA)
        freqA.create_dataset("listOfTxPolarizations", data=numpy.string_([txP]),
            maxshape=(2,))
    else:
        freqA = fid[freqA]

    txPolList = freqA["listOfTxPolarizations"]
    if not numpy.string_(txP) in txPolList:
        assert len(txPolList) == 1
        txPolList.resize((2,))
        txPolList[1] = txP

    ##Now add in transmit specific information
    txgrpstr = '/science/LSAR/RRSD/swaths/frequencyA/tx{0}'.format(txP)
    firstInPol = False
    if txgrpstr not in fid:
        firstInPol = True
        tstart = datetime.datetime(firstrec.SensorAcquisitionYear, 1, 1) +\
                 datetime.timedelta(days=int(firstrec.SensorAcquisitionDayOfYear-1))
        txgrp = fid.create_group(txgrpstr)
        time = txgrp.create_dataset('UTCtime', dtype='f8', shape=(nLines,))
        time.attrs['units'] = numpy.bytes_("seconds since {0}T00:00:00".format(tstart.strftime('%Y-%m-%d')))
        txgrp.create_dataset('numberOfSubSwaths', data=len(sub_swaths))
        txgrp.create_dataset('radarTime', dtype='f8', shape=(nLines,))
        txgrp.create_dataset('rangeLineIndex', dtype='i8', shape=(nLines,))
        for i in range(len(sub_swaths)):
            key = f"validSamplesSubSwath{i + 1}"
            txgrp.create_dataset(key, dtype='i8', shape=(nLines, 2))
        txgrp.create_dataset('centerFrequency', data=speed_of_light / (ldr.summary.RadarWavelengthInm))
        txgrp.create_dataset('rangeBandwidth', data=ldr.calibration.header.BandwidthInMHz * 1.0e6)
        txgrp.create_dataset('chirpDuration', data=firstrec.ChirpLengthInns * 1.0e-9)
        txgrp.create_dataset('chirpSlope', data=-((txgrp['rangeBandwidth'][()])/(txgrp['chirpDuration'][()])))
        txgrp.create_dataset('nominalAcquisitionPRF', data=prf)
        txgrp.create_dataset('slantRangeSpacing', data=dr)
        txgrp.create_dataset('slantRange', data=r0 + numpy.arange(nPixels) * dr)
        txgrp.create_dataset('listOfTxTRMs', data=numpy.asarray([1], dtype='uint8'))
        txgrp.create_dataset('sceneCenterAlongTrackSpacing', data=ds)
        txgrp.create_dataset('sceneCenterGroundRangeSpacing', data=dg)
        # dummy calibration data
        txgrp.create_dataset('txPhase', data=numpy.zeros((nLines,1), dtype='f4'))
        txgrp.create_dataset('chirpCorrelator', data=numpy.ones((nLines,1,3), dtype='c8'))
        txgrp.create_dataset('calType', data=makeDummyCalType(nLines))
    else:
        txgrp = fid[txgrpstr]

    ###Create imagery group
    rximgstr = os.path.join(txgrpstr, 'rx{0}'.format(rxP))
    if rximgstr in fid:
        fid.close()
        raise ValueError('Reparsing polarization {0}. Array already exists {1}'.format(pol, rximgstr))

    print('Dimensions: {0}L x {1}P'.format(nLines, nPixels))
    rxgrp = fid.create_group(rximgstr)

    # Dummy cal data
    rxgrp.create_dataset('caltone', data=numpy.ones((nLines,1), dtype='c8'))
    rxgrp.create_dataset('attenuation', data=numpy.ones((nLines,1), dtype='f4'))
    rxgrp.create_dataset('TRMDataWindow', data=numpy.ones((nLines,1), dtype='u1'))
    # Create List of RX TRMs
    rxgrp.create_dataset('listOfRxTRMs', data=numpy.asarray([1], dtype='uint8'))

    ##Set up BFPQLUT
    assert firstrec.SARRawSignalData.dtype.itemsize <= 2
    lut = numpy.arange(2**16, dtype=numpy.float32)
    assert numpy.issubdtype(firstrec.SARRawSignalData.dtype, numpy.signedinteger)
    lut[-2**15:] -= 2**16
    assert ldr.summary.DCBiasIComponent == ldr.summary.DCBiasQComponent
    lut -= ldr.summary.DCBiasIComponent
    BAD_VALUE = 2**15
    lut[BAD_VALUE] = 0
    rxlut = fid.create_dataset(os.path.join(rximgstr, 'BFPQLUT'), data=lut)


    #Create imagery layer
    compress = dict(chunks=(4, 512), compression="gzip", compression_opts=9, shuffle=True)
    cpxtype = numpy.dtype([('r', numpy.uint16), ('i', numpy.uint16)])
    rximg = fid.create_dataset(os.path.join(rximgstr, pol), dtype=cpxtype, shape=(nLines,nPixels), **compress)
    # Per http://cfconventions.org/Data/cf-conventions/cf-conventions-1.9/cf-conventions.html#missing-data
    rximg.attrs['_FillValue'] = BAD_VALUE

    ##Start populating the imagery

    rec = firstrec
    for linnum in range(1, nLines+1):
        if (linnum % 1000 == 0):
            print('Parsing Line number: {0} out of {1}'.format(linnum, nLines))

        if firstInPol:
            tx_radar_time = rec.SensorAcquisitionmsecsOfDay * 1.0e-3 - dwp_delay 
            txgrp['UTCtime'][linnum-1] = tx_radar_time
            txgrp['rangeLineIndex'][linnum-1] = rec.SARImageDataLineNumber
            txgrp['radarTime'][linnum-1] = tx_radar_time

        #Adjust range line
        rshift = int(numpy.rint((rec.SlantRangeToFirstSampleInm - r0) / dr))
        write_arr = numpy.full((2*nPixels), BAD_VALUE, dtype=numpy.uint16)

        inarr = rec.SARRawSignalData[0,:].astype(numpy.uint16)

        left = 2 * rec.ActualCountOfLeftFillPixels
        right = 2 * rec.ActualCountOfRightFillPixels
        inarr[:left] = BAD_VALUE
        inarr[-right:] = BAD_VALUE

        if rshift >= 0:
            write_arr[2*rshift:] = inarr[:2*(nPixels - rshift)]
        else:
            write_arr[:2*rshift] = inarr[-2*rshift:]

        for i in range(len(sub_swaths) - 1):
            gap_start, gap_stop = 2 * sub_swaths[i][1], 2 * sub_swaths[i + 1][0]
            write_arr[gap_start:gap_stop] = BAD_VALUE

        if firstInPol:
            # check if any samples at the very start of RX window are missing.
            inds = numpy.where(write_arr != BAD_VALUE)[0]
            if (len(inds) > 0 and inds[0] > 0):
                warn(f'The first {inds[0] // 2} range samples are missing. '
                      f'They are filled with {BAD_VALUE}!')
            for i in range(len(sub_swaths)):
                key = f"validSamplesSubSwath{i + 1}"
                txgrp[key][linnum-1] = sub_swaths[i]

        #Complex float 16 writes work with write_direct only
        rximg.write_direct(write_arr.view(cpxtype), dest_sel=numpy.s_[linnum-1])

        ##Read next record
        if linnum != nLines:
            rec = image.readNextLine()

    if firstInPol:
        #Adjust time records - ALOS provides this only to nearest millisec - not good enough
        tinp = txgrp['UTCtime'][:]
        prf = txgrp['nominalAcquisitionPRF'][()]
        tarr = (tinp - tinp[0]) * 1000
        ref = numpy.arange(tinp.size) / prf

        ####Check every 20 microsecs
        off = numpy.arange(-50,50)*2.0e-5
        res = numpy.zeros(off.size)

        ###Check which offset produces the same millisec truncation
        ###Assumes PRF is correct
        for xx in range(off.size):
            ttrunc = numpy.floor((ref+off[xx])*1000) #Should be round / floor?
            res[xx] = numpy.sum(tarr-ttrunc)

        delta = (numpy.argmin(numpy.abs(res)) - 50)*2.0e-5
        print('Start time correction in usec: ', delta*1e6)
        txgrp['UTCtime'][:] = ref + tinp[0] + delta


def computeBoundingPolygon(h5file: str, h: float = 0.0):
    """Compute bounding polygon given (an otherwise complete) NISAR L0B product.
    Uses a fixed height in lieu of a digital elevation model.
    """
    dem = isce3.geometry.DEMInterpolator(h)
    raw = Raw(hdf5file=h5file)
    orbit = raw.getOrbit()
    _, grid = raw.getRadarGrid()
    fc, doppler = make_doppler_lut([h5file], az=0.0)
    # Make sure we don't accidentally have an inconsistent wavelength.
    assert numpy.isclose(grid.wavelength, isce3.core.speed_of_light / fc)
    return isce3.geometry.get_geo_perimeter_wkt(grid, orbit, doppler, dem)


def finalizeIdentification(h5file: str):
    """Add identification fields that depend on swath information:
    {"boundingPolygon"," zeroDopplerStartTime", "zeroDopplerEndTime"}
    """
    # NOTE Need complete product to use product reader class, so assume we've
    # already populated dummy values.
    epoch, t = Raw(hdf5file=h5file).getPulseTimes()
    t0 = (epoch + isce3.core.TimeDelta(t[0])).isoformat()
    t1 = (epoch + isce3.core.TimeDelta(t[-1])).isoformat()
    poly = computeBoundingPolygon(h5file)

    with h5py.File(h5file, 'r+') as fid:
        ident = fid["/science/LSAR/identification"]
        # Unlink datasets and create new ones to avoid silent truncation.
        del ident["boundingPolygon"]
        del ident["zeroDopplerStartTime"]
        del ident["zeroDopplerEndTime"]
        def additem(name, value, description):
            ds = ident.create_dataset(name, data=numpy.string_(value))
            ds.attrs["description"] = numpy.string_(description)
        additem("boundingPolygon", poly, descriptions.bounding_polygon)
        additem("zeroDopplerStartTime", t0, "Azimuth start time of product")
        additem("zeroDopplerEndTime", t1, "Azimuth stop time of product")


def process(args=None):
    '''
    Main processing workflow.
    '''
    #Discover file names
    filenames = getALOSFilenames(args.indir)

    #Parse the leader file
    leader = parseLeaderFile(filenames)

    #Set up output file name
    if args.outh5 is None:
        args.outh5 = filenames['defaulth5']

    if os.path.exists(args.outh5):
        raise ValueError('Output HDF5 file {0} already exists. Exiting ...'.format(args.outh5))


    #Setup HDF5 skeleton
    constructNISARHDF5(args, leader)


    #Iterate over polarizations for imagery layers
    for pol in ['HH', 'HV', 'VV', 'VH']:
        if pol in filenames:
            addImagery(args.outh5, leader, filenames[pol], pol,
                args.gap_width_usec, args.gap_location)

    finalizeIdentification(args.outh5)


if __name__ == "__main__":
    '''
    Main driver.
    '''

    #Parse command line
    inps = cmdLineParse()

    #Process the data
    process(args=inps)
