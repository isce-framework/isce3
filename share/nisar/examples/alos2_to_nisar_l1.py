#!/usr/bin/env python3

import os
import numpy as np
import glob
import argparse
import h5py
import datetime
from dateutil.parser import isoparse
import time

import isce3
from alos_to_nisar_l0b import (get_alos_orbit, set_h5_orbit, getset_attitude,
                               ident_descriptions)
from isce3.stripmap.readers.l1.ALOS2.CEOS import ImageFile
from isce3.product import RadarGridParameters
from isce3.core import DateTime

'''
References:

https://en.wikipedia.org/wiki/Earth_radius#Directional
https://books.google.com/books?id=pFO6VB_czRYC&pg=PA98#v=onepage&q&f=false
https://www.eorc.jaxa.jp/ALOS-2/en/doc/fdata/PALSAR-2_xx_Format_CEOS_E_f.pdf

'''

FLAG_REFORMAT_DOPPLER_ISCE2 = True
SPEED_OF_LIGHT = 299792458.0
ALL_POLARIZATIONS_SET = set(['HH', 'HV', 'VV', 'VH', 'RH', 'RV'])

CALIBRATION_FIELD_DICT = {
    'elevationAntennaPattern': 'Complex two-way elevation antenna pattern',
    'nes0': 'Noise equivalent sigma zero'
}


def parse_args():
    '''
    Command line parser.
    '''
    parser = argparse.ArgumentParser(
        description="Package ALOS-2 L1 stripmap data into NISAR L1 RSLC HDF5")
    parser.add_argument('-i', '--indir', dest='indir', type=str,
                        help="Folder containing one ALOS-2 L1 module",
                        required=True)
    parser.add_argument('-l', '--pols',
                        dest='polarization_list', type=str,
                        help="List of polarizations to process",
                        nargs='*')
    parser.add_argument('--first-line',
                        dest='first_line', type=int,
                        help="First azimuth line to unpack")
    parser.add_argument('--last-line',
                        dest='last_line', type=int,
                        help="Last azimuth line to unpack")
    parser.add_argument('-o', '--outh5', dest='outh5', type=str,
                        help="Name of output file. If not provided, will be"
                             " determined from ALOS-2 granule")

    parser_verbose = parser.add_mutually_exclusive_group()
    parser_verbose.add_argument('-q',
                                '--quiet',
                                dest='verbose',
                                action='store_false',
                                help='Activate quiet (non-verbose) mode',
                                default=True)
    parser_verbose.add_argument('-v',
                                '--verbose',
                                dest='verbose',
                                action='store_true',
                                help='Activate verbose mode',
                                default=True)

    args = parser.parse_args()
    if not os.path.isdir(args.indir):
        raise ValueError(
            '{0} does not appear to be a directory'.format(args.indir))

    if args.outh5 is None:
        print('HDF5 output granule name will be determined on fly and created'
              ' in cwd')

    return args


def process(args=None):
    '''
    Main processing workflow.
    '''
    start_time = time.time()
    if not args.polarization_list:
        args.polarization_list = ['HH', 'HV', 'VV', 'VH']

    # Discover file names
    filenames = get_alos_filenames(args.indir, args)

    # Parse the leader file
    leader = parse_leader_file(filenames, args)

    # Set up output file name
    if args.outh5 is None:
        args.outh5 = filenames['defaulth5']

    if os.path.exists(args.outh5):
        raise ValueError(f'Output HDF5 file {args.outh5} already exists.'
                         ' Exiting ...')

    # Setup HDF5 skeleton
    orbit = construct_nisar_hdf5(args.outh5, leader)

    # Iterate over polarizations for imagery layers
    metadata = {}

    pol_list = []
    for count, pol in enumerate(args.polarization_list):
        pol_upper = pol.upper()
        if pol_upper not in filenames:
            continue
        add_imagery(args, leader, filenames[pol_upper], pol_upper, orbit,
                    metadata, filenames, flag_first_image=count == 0)
        pol_list.append(pol_upper)

    populate_hdf5(metadata, args.outh5, orbit, pol_list)

    print('saved file:', args.outh5)

    elapsed_time = time.time() - start_time
    hms_str = str(datetime.timedelta(seconds=int(elapsed_time)))
    print(f'elapsed time: {hms_str}s ({elapsed_time:.3f}s)')


def get_alos_filenames(indir, args):
    '''
    Parse the contents of a given directory to separate out leader and image
    files.
    '''

    filenames = {}

    # First look for the leader file
    flist = glob.glob(os.path.join(indir, 'LED-ALOS2*1.1__*'))
    if len(flist) == 0:
        raise ValueError('No leader files found in folder {0}'.format(indir))
    elif len(flist) > 1:
        raise ValueError('Multiple leader files in folder {0}'.format(indir))

    filenames['leaderfile'] = flist[0]
    pattern = os.path.basename(flist[0])[4:]

    # Look for polarizations
    if args.verbose:
        print('looking for available polarizations...')
    for pol in args.polarization_list:
        flist = glob.glob(os.path.join(
            indir, 'IMG-{0}-{1}'.format(pol, pattern)))
        if len(flist) == 1:
            if args.verbose:
                print('    found polarization: {0}'.format(pol))
            filenames[pol] = flist[0]

    # If no image files were found
    if len(filenames) == 1:
        raise ValueError(
            'No image files were found in folder: {0}'.format(indir))

    filenames['defaulth5'] = '{0}.h5'.format(pattern)
    return filenames


def parse_leader_file(filenames, args):
    '''
    Parse leader file and check values against polarizations.
    '''

    from isce3.stripmap.readers.l1.ALOS2.CEOS import LeaderFile
    try:
        ldr = LeaderFile.LeaderFile(filenames['leaderfile'])
    except AssertionError as msg:
        print(msg)
        raise AssertionError(
            'Error parsing ALOS-2 L1.1 leader file: {0}'.format(
                filenames['leaderfile']))

    # Checks to ensure that the number of polarizations is consistent
    numpol = len(filenames) - 2  # Subtract leader and defaulth5 name

    if (not args.polarization_list and
            numpol != ldr.summary.NumberOfSARChannels):
        print(f'WARNING Number of image files ({numpol}) discovered'
              ' is inconsistent with Leader File'
              f' ({ldr.summary.NumberOfSARChannels})')

    return ldr


def construct_nisar_hdf5(outh5, ldr):
    '''
    Build skeleton of HDF5 file using leader file information.
    '''

    # Open file for writing
    root_group = h5py.File(outh5, 'w')
    lsar_group = root_group.create_group('/science/LSAR')
    ident_group = lsar_group.create_group('identification')

    # scalar
    ident_group.create_dataset('diagnosticModeFlag', data=np.uint8(0))
    ident_group.create_dataset('isGeocoded', data=np.bytes_("False"))
    ident_group.create_dataset('listOfFrequencies', data=np.bytes_(["A"]))
    ident_group.create_dataset('missionId', data=np.bytes_("ALOS-2"))
    if ldr.summary.TimeDirectionIndicatorAlongLine[0] == "A":
        direction = "ascending"
    else:
        direction = "descending"
    ident_group.create_dataset('orbitPassDirection',
                               data=np.bytes_(direction))
    ident_group.create_dataset('processingType',
                               data=np.bytes_("repackaging"))
    ident_group.create_dataset('productType', data=np.bytes_("RSLC"))
    ident_group.create_dataset('productVersion', data=np.bytes_("0.1.0"))
    ident_group.create_dataset('absoluteOrbitNumber',
                               data=np.array(0, dtype='u4'))
    ident_group.create_dataset('trackNumber', data=np.array(0, dtype=np.uint8))
    ident_group.create_dataset('frameNumber', data=np.array(0,
                                                            dtype=np.uint16))
    ident_group.create_dataset("isUrgentObservation", data=np.bytes_("False"))

    ident_group.create_dataset("plannedObservationId", data=np.bytes_(["0"]))
    # shape = numberOfDatatakes
    ident_group.create_dataset("plannedDatatakeId", data=np.bytes_(["0"]))

    # fields added to spec in 2023
    ident_group.create_dataset("granuleId", data=np.bytes_("None"))
    ident_group.create_dataset("instrumentName", data=np.bytes_("PALSAR-2"))
    ident_group.create_dataset("isDithered", data=np.bytes_("False"))
    ident_group.create_dataset("isMixedMode", data=np.bytes_("False"))
    ident_group.create_dataset("processingCenter",
                               data=np.bytes_(
                                   "JAXA (SLC repackaged at JPL)"))
    ident_group.create_dataset(
        "processingDateTime",
        data=np.bytes_(datetime.datetime.now().isoformat()))
    ident_group.create_dataset("productLevel", data=np.bytes_("L1"))
    ident_group.create_dataset(
        "productSpecificationVersion",
        data=np.bytes_("0.9.0"))
    ident_group.create_dataset("radarBand", data=np.bytes_("L"))

    # Start populating metadata parts
    rslc = lsar_group.create_group('RSLC')
    rslc.create_group('metadata/processingInformation/inputs')

    # Start populating metadata
    orbit_group = rslc.create_group('metadata/orbit')
    attitude_group = rslc.create_group('metadata/attitude')
    orbit = get_alos_orbit(ldr)
    set_h5_orbit(orbit_group, orbit)
    getset_attitude(attitude_group, ldr, orbit)

    return orbit


def add_imagery(args, ldr, imgfile, pol, orbit, metadata, filenames,
                flag_first_image):
    '''
    Populate swaths segment of HDF5 file.
    '''

    verbose = args.verbose

    root_group = h5py.File(args.outh5, 'r+')
    assert len(pol) == 2

    # parse imagefile descriptor and first record.
    image = ImageFile.ImageFile(imgfile)
    firstrec = image.readNextLine()

    # set range-grid parameters
    fsamp = ldr.summary.SamplingRateInMHz * 1.0e6
    r0 = firstrec.SlantRangeToFirstSampleInm
    dr = SPEED_OF_LIGHT / (2 * fsamp)
    da = ldr.summary.LineSpacingInm
    bytesperpixel = (image.description.NumberOfBytesPerDataGroup //
                     image.description.NumberOfSamplesPerDataGroup)
    width = (image.description.NumberOfBytesOfSARDataPerRecord //
             bytesperpixel) // image.description.NumberOfSamplesPerDataGroup
    length = image.description.NumberOfSARDataRecords

    freq_str = '/science/LSAR/RSLC/swaths/frequencyA'

    calibration_factor_db = ldr.calibration.header.CalibrationFactor - 32
    calibration_factor = np.sqrt(10.0**(calibration_factor_db/10))

    if verbose:
        print('absolute radiometric correction (DN to sigma-naught)')
        print('    calibration factor [dB]:', calibration_factor_db)
        print('    calibration factor [linear]:', calibration_factor)

    # If this is first pol being written, add common information as well
    if flag_first_image:
        freq_group = root_group.create_group(freq_str)
        wavelength = ldr.summary.RadarWavelengthInm
        freq_group.create_dataset('centerFrequency',
                                  data=SPEED_OF_LIGHT / wavelength)

        bandwidth = ldr.summary.TotalProcessorBandwidthInRange * 1.0e3
        freq_group.create_dataset('rangeBandwidth', data=bandwidth)

        freq_group.create_dataset('chirpDuration',
                                  data=firstrec.ChirpLengthInns * 1.0e-9)
        freq_group.create_dataset(
            'chirpSlope',
            data=-((freq_group['rangeBandwidth'][()]) /
                   (freq_group['chirpDuration'][()])))

        # The variable `ldr.summary.NominalPRFInmHz` has more significant digits
        # but may not be more correct
        # prf = ldr.summary.NominalPRFInmHz * 1.0e-3
        prf = firstrec.PRFInmHz * 1.0e-3

        freq_group.create_dataset('nominalAcquisitionPRF', data=prf)

        assert (ldr.summary.SensorIDAndMode[7] == 'R' or
                ldr.summary.SensorIDAndMode[7] == 'L')

        operation_mode_number = ldr.summary.SensorIDAndMode[10:12]
        # NESZ values from (slide 7):
        # https://www.eorc.jaxa.jp/ALOS-2/en/about/palsar2.htm
        if operation_mode_number == '00':
            operation_mode = "Spotlight mode"
            nesz = -24
        elif operation_mode_number == '01':
            operation_mode = "Ultra-fine mode"
            nesz = -24
        elif operation_mode_number == '02':
            operation_mode = "High-sensitive mode"
            nesz = -28
        elif operation_mode_number == '03':
            operation_mode = "Fine mode"
            nesz = -26
        elif operation_mode_number == '08':
            operation_mode = "ScanSAR nominal mode"
            nesz = -26
        elif (operation_mode_number == '09' and
                int(np.round(bandwidth/1e6)) == 14):
            operation_mode = "ScanSAR wide mode (14 MHz)"
            nesz = -26
        elif (operation_mode_number == '09' and
                int(np.round(bandwidth/1e6)) == 28):
            operation_mode = "ScanSAR wide mode (28 MHz)"
            nesz = -23
        elif operation_mode_number == '18':
            operation_mode = "Full (Quad.) pol./High-sensitive mode"
            nesz = -25
        elif operation_mode_number == '19':
            operation_mode = "Full (Quad.) pol./Fine mode"
            nesz = -23
        elif operation_mode_number == '64':
            operation_mode = "Manual observation"
            nesz = -24
        else:
            print('WARNING unknown operation mode:', operation_mode_number)
            operation_mode = "Unknown"

        metadata['Operation Mode'] = operation_mode
        metadata['NESZ'] = nesz

        metadata['Center Wavelength'] = wavelength
        metadata['Bandwidth'] = bandwidth
        metadata['Average Pulse Repetition Interval'] = 1.0 / prf
        metadata['Azimuth Spacing per Bin'] = da
        metadata['Effective Velocity'] = da * prf

        leader_fie = filenames['leaderfile']
        product_id = leader_fie.split('-')[3]

        if product_id[3] == 'L':
            lookside = 'left'
        else:
            lookside = 'right'
        metadata['Look Direction'] = lookside.upper()

        if verbose:
            print('parameters from metadata:')
            print(f'    operation mode: {operation_mode}'
                  f' ({operation_mode_number})')
            print('    product ID:', product_id)
            print('    bandwidth:', bandwidth)
            print('    prf: ', prf)
            print('    azimuth spacing: ', da)
            print('    effective velocity: ', da * prf)
            print('    look direction:', lookside)

        freq_group.create_dataset('slantRangeSpacing', data=dr)
        freq_group.create_dataset('slantRange',
                                  data=r0 + np.arange(width) * dr)

        if not FLAG_REFORMAT_DOPPLER_ISCE2:
            doppler_coeffs = [
                ldr.summary.CrossTrackDopplerConstantTermInHz,
                ldr.summary.CrossTrackDopplerLinearTermInHzPerPixel,
                ldr.summary.CrossTrackDopplerLinearTermInHzPerPixel2]
            metadata['Doppler coeffs km'] = doppler_coeffs
            if verbose:
                print('    Doppler coeffs: [km]',
                      ', '.join(map(str, doppler_coeffs)))
        else:
            doppler_coeffs = [ldr.summary.DopplerCenterFrequencyConstantTerm,
                              ldr.summary.DopplerCenterFrequencyLinearTerm]
            rng = r0 + np.arange(0, width, 100) * dr
            doppler = doppler_coeffs[0] + doppler_coeffs[1] * rng / 1000
            dfit = np.polyfit(np.arange(0, width, 100), doppler, 1)
            doppler_coeffs_rbin = [dfit[1], dfit[0], 0., 0.]
            metadata['Doppler coeffs rbin'] = doppler_coeffs_rbin
            if verbose:
                print('    Doppler coeffs [rbin/index]:',
                      ', '.join(map(str, doppler_coeffs)))

        azfmrate_coeff = [
            ldr.summary.CrossTrackDopplerRateConstantTermInHzPerSec,
            ldr.summary.CrossTrackDopplerRateLinearTermInHzPerSecPerPixel,
            ldr.summary.CrossTrackDopplerRateQuadraticTermInHzPerSecPerPixel2]

        metadata['Azimuth FM rate'] = azfmrate_coeff
        if verbose:
            print('    azimuth FM rate coeffs:', azfmrate_coeff)

        sensing_start = \
            (datetime.datetime(firstrec.SensorAcquisitionYear, 1, 1) +
             datetime.timedelta(
                 days=int(firstrec.SensorAcquisitionDayOfYear-1),
                 seconds=firstrec.SensorAcquisitionusecsOfDay*1e-6))

        freq_group.create_dataset('numberOfSubSwaths', data=1, dtype='i8')
        freq_group.create_dataset('validSamplesSubSwath1', dtype='i8',
                                  data=np.tile([0, width], (length, 1)))

        metadata['Mission'] = 'ALOS-2'
        metadata['Image Starting Range'] = r0
        metadata['Range Spacing per Bin'] = dr
        metadata['SLC width'] = width
        metadata['SLC length'] = length

    BAD_VALUE = -2**15

    # Create imagery layer
    compress = dict(chunks=(4, 512), compression="gzip",
                    compression_opts=9, shuffle=True)
    cpxtype = np.dtype([('r', np.float32), ('i', np.float32)])
    polimg = root_group.create_dataset(os.path.join(
        freq_str, pol), dtype=cpxtype, shape=(length, width), **compress)

    # Start populating the imagery
    rec = firstrec
    if args.first_line is not None:
        first_line = args.first_line
    else:
        first_line = 1
    if args.last_line is not None:
        last_line = min([args.last_line, length+1])
    else:
        last_line = length+1

    print(f'processing polarization {pol} ({length}L x {width}P):')
    for linnum in range(first_line, last_line):

        if (linnum % 1000 == 0):
            print('    line number: {0} out of {1}'.format(linnum, length))

        # Adjust range line
        rshift = int(np.rint((rec.SlantRangeToFirstSampleInm - r0) / dr))
        write_arr = np.full((2 * width), BAD_VALUE, dtype=np.float32)

        inarr = rec.SARRawSignalData[0, :]

        if rshift >= 0:
            write_arr[2*rshift:] = inarr[:2 * (width - rshift)]
        else:
            write_arr[:2*rshift] = inarr[-2 * rshift:]

        # Apply absolute radiometric correction
        write_arr *= calibration_factor

        # Complex float 16 writes work with write_direct only
        polimg.write_direct(write_arr.view(cpxtype), dest_sel=np.s_[linnum-1])

        # Read next record
        if linnum != length:
            rec = image.readNextLine()

    if flag_first_image:
        sensing_end = sensing_start + datetime.timedelta(seconds=length / prf)

        # Get azimuth time bounds of the scene
        metadata['Start Time of Acquisition'] = sensing_start
        metadata['Stop Time of Acquisition'] = sensing_end

        sensing_mid = sensing_start + (sensing_end - sensing_start) / 2
        ref_epoch = orbit.reference_epoch.isoformat()
        if verbose:
            print('time parameters:')
            print('    start: ', sensing_start)
            print('    mid: ', sensing_mid)
            print('    end: ', sensing_end)
            print('    reference epoch: ', ref_epoch)

        metadata['Scene Center Incidence Angle'] = \
            ldr.summary.SceneCenterIncidenceAngle

        ref_epoch = DateTime(ref_epoch)
        timedelta_start = (DateTime(sensing_start) -
                           ref_epoch).total_seconds()
        radar_grid = RadarGridParameters(timedelta_start,
                                         wavelength,
                                         prf,
                                         r0,
                                         dr,
                                         lookside,
                                         length,
                                         width,
                                         ref_epoch)
        metadata['Radar Grid'] = radar_grid


def populate_hdf5(metadata, outfile, orbit, pol_list, frequency='A'):
    """
    Generate a Level-1 NISAR format HDF5 product.
    """

    # Generate a common azimuth time vs. slant range grid for all calibration
    # grids. This also sets the reference epoch used for all subsequent
    # dataset generation
    construct_calibration_grid(metadata, orbit)

    # Now open it for modification
    with h5py.File(outfile, 'r+') as root_group:

        # Set global CF conventions attribute
        if frequency == 'A':
            root_group.attrs['Conventions'] = np.bytes_('CF-1.7')

        # Update the calibration information
        update_calibration_information(root_group, metadata, pol_list,
                                       frequency)

        # Update the Dopplers
        update_doppler(root_group, metadata, frequency)

        # Update the radar metadata
        update_metadata(root_group, metadata, pol_list, frequency=frequency)

        # Update identification
        update_identification(root_group, orbit, metadata)


def update_metadata(fid, metadata, pol_list, frequency='A'):
    """
    Update radar metadata. This function mainly interfaces with the
    science/LSAR/RSLC/swaths group to set the right scalar parameters.
    """
    # Open the correct frequency swath group
    group = fid['science/LSAR/RSLC/swaths/frequency' + frequency]

    # Update polarization list
    group['listOfPolarizations'] = np.array(pol_list, dtype='S2')
    group['listOfPolarizations'].attrs['description'] = np.bytes_(
        'List of processed polarization layers with frequency ' + frequency)

    # Create new slant range array for all pixels
    del group['slantRange']
    R = (metadata['Image Starting Range'] + metadata['Range Spacing per Bin'] *
         np.arange(metadata['SLC width']))
    group['slantRange'] = R
    group['slantRange'].attrs['description'] = np.bytes_(
        'CF compliant dimension associated with slant range'
    )
    group['slantRange'].attrs['units'] = np.bytes_('meters')
    group['slantRangeSpacing'][...] = metadata['Range Spacing per Bin']

    inc = np.radians(metadata['Scene Center Incidence Angle'])

    group['sceneCenterGroundRangeSpacing'] = metadata['Range Spacing per Bin'] / np.sin(inc)

    # Bandwidth data
    group['acquiredRangeBandwidth'] = metadata['Bandwidth']
    group['processedRangeBandwidth'] = metadata['Bandwidth']

    # Center frequency
    group['acquiredCenterFrequency'] = (SPEED_OF_LIGHT /
                                        metadata['Center Wavelength'])
    group['processedCenterFrequency'] = (SPEED_OF_LIGHT /
                                         metadata['Center Wavelength'])

    # Nominal PRF
    group['nominalAcquisitionPRF'][...] = \
        1.0 / metadata['Average Pulse Repetition Interval']

    # Azimuth pixel spacing
    group['sceneCenterAlongTrackSpacing'] = metadata['Azimuth Spacing per Bin']

    # Azimuth bandwidth
    if 'Azimuth Spacing per Bin' in metadata.keys():
        azres = metadata['Azimuth Spacing per Bin']
        group['processedAzimuthBandwidth'] = (metadata['Effective Velocity'] /
                                              (2.0 * azres))

    elif 'Antenna Length' in metadata.keys():
        azres = 0.6 * metadata['Antenna Length']
        group['processedAzimuthBandwidth'] = (metadata['Effective Velocity'] /
                                              (2.0 * azres))

    # Create array of azimuth times
    if frequency == 'A':
        group = fid['science/LSAR/RSLC/swaths']
        pri = metadata['Average Pulse Repetition Interval']
        ref_epoch = metadata['ref_epoch']
        t0 = (metadata['Start Time of Acquisition'] - ref_epoch).total_seconds()
        t = t0 + pri * np.arange(metadata['SLC length'])
        if 'zeroDopplerTime' in group:
            desc = group['zeroDopplerTime'].attrs['description']
            del group['zeroDopplerTime']
        else:
            desc = ''
        group['zeroDopplerTime'] = t
        group['zeroDopplerTime'].attrs['description'] = desc
        group['zeroDopplerTime'].attrs['units'] = np.bytes_(
            metadata['ref_epoch_attr'])
        group['zeroDopplerTimeSpacing'] = pri


def construct_calibration_grid(metadata, orbit, az_pad_in_pixels=20,
                               rg_pad_in_pixels=20):
    """
    Construct a low-resolution azimuth time vs. slant range grid to be used
    for all calibration and geolocation grids. Spacing hard-coded for different
    sensors. This function needs to be generalized to adapt to various
    topography heights and platform altitudes.

    Parameters
    ----------
    metadata: dict
        Dictionary containing metadata information
    orbit: isce3.core.Orbit
        ISCE3 orbit object
    az_pad_in_pixels: scalar
        Azimuth pad in pixels
    rg_pad_in_pixels: scalar
        Slant-range pad in pixels
    """
    # Set calibration grid spacing
    rg_spacing = 1000
    az_spacing = 0.25

    # Set slant-range bounds. Extend range stop beyond SLC width to ensure
    # SLC fully contained within calibration grid.
    r_start = metadata['Image Starting Range'] - rg_pad_in_pixels * rg_spacing
    r_stop = (r_start + (metadata['SLC width'] + 1) * rg_spacing +
              rg_pad_in_pixels * rg_spacing)

    # Get azimuth time bounds of the scene
    a0 = metadata['Start Time of Acquisition']
    a1 = metadata['Stop Time of Acquisition']

    # Convert azimuth dates to seconds
    ref_epoch_iso_format = orbit.reference_epoch.isoformat()
    ref_epoch = isoparse(ref_epoch_iso_format)
    metadata['ref_epoch'] = ref_epoch
    metadata['Start Seconds of Acquisition'] = (a0 - ref_epoch).total_seconds()
    metadata['Stop Seconds of Acquisition'] = (a1 - ref_epoch).total_seconds()
    metadata['ref_epoch_attr'] = 'seconds since %s' % ref_epoch.isoformat(
        sep=' ')

    # Pad the azimuth time bounds in each direction (az_pad in units of
    # seconds)
    a0 = round((a0 - ref_epoch).total_seconds() -
               az_pad_in_pixels * az_spacing)
    a1 = round((a1 - ref_epoch).total_seconds() +
               az_pad_in_pixels * az_spacing)

    # Construct grids and update metadata dictionary
    rgrid = np.arange(r_start, r_stop, rg_spacing, dtype=np.float64)
    agrid = np.arange(a0, a1, az_spacing, dtype=np.float64)
    metadata['calibration_range_grid'] = rgrid
    metadata['calibration_azimuth_grid'] = agrid


def update_identification(fid, orbit, metadata):
    """
    Updates the science/LSAR/identification group.
    """
    group = fid['science/LSAR/identification']

    # Zero doppler times
    start = metadata['Start Time of Acquisition']
    stop = metadata['Stop Time of Acquisition']

    group.create_dataset('zeroDopplerStartTime',
                         data=np.bytes_(start.isoformat()))

    group.create_dataset('zeroDopplerEndTime',
                         data=np.bytes_(stop.isoformat()))

    # Look direction
    group.create_dataset('lookDirection',
                         data=np.bytes_(metadata['Look Direction'].title()))

    # Radar grid
    radar_grid = metadata['Radar Grid']

    # Compute the bounding polygon at a fixed 0 m height
    height = 0
    dem = isce3.geometry.DEMInterpolator(height)
    doppler = isce3.core.LUT2d()
    poly = isce3.geometry.get_geo_perimeter_wkt(radar_grid, orbit, doppler,
                                                dem)

    # Allocate bounding polygon in the identification group
    group.create_dataset('boundingPolygon', data=np.bytes_(poly))
    group['boundingPolygon'].attrs['epsg'] = 4326
    group['boundingPolygon'].attrs['ogr_geometry'] = np.bytes_('polygon')

    for name, desc in ident_descriptions.items():
        group[name].attrs["description"] = np.bytes_(desc)


def _create_lut_coordinate_vectors(h5_group, zero_doppler_time_vector,
                                   slantrange_vector, description,
                                   time_units):
    """
    Create look-up table LUT coordinate vectors "slantRange" and
    "zeroDopplerTime"

    Parameters
    ----------
    h5_group: h5py object
        H5 group that will hold the LUT coordinate vectors
    zero_doppler_time_vector: np.ndarray
        Zero-Doppler time vector
    slantrange_vector: : np.ndarray
        Slant-range vector
    description: str
        LUT description
    time_units: str
        Time units
    """

    if 'slantRange' not in h5_group:
        h5_group.create_dataset(
                'slantRange', data=slantrange_vector)
        h5_group['slantRange'].attrs['description'] = np.bytes_(
                'Slant range dimension corresponding to'
                f' {description} records')
        h5_group['slantRange'].attrs['units'] = np.bytes_('meters')

    if 'zeroDopplerTime' not in h5_group:
        h5_group.create_dataset(
                'zeroDopplerTime', data=zero_doppler_time_vector)
        h5_group['zeroDopplerTime'].attrs['description'] = np.bytes_(
                'Zero doppler time dimension corresponding to'
                f' {description} records')
        h5_group['zeroDopplerTime'].attrs['units'] = np.bytes_(time_units)


def update_calibration_information(fid, metadata, pol_list, frequency):
    """
    Fill calibration information LUTs (nes0 and elevation antenna pattern)
    with zeros,

    Parameters
    ----------
    fid: h5py object
        H5 group that will hold the LUT coordinate vectors
    metadata:
        Dictionary containing metadata information
    pol_list: list(str)
        List of polarizations for given frequency
    frequency: str
        Frequency band (e.g., "A" or "B")
    """
    # Get doppler group from metadata
    parameters = 'science/LSAR/RSLC/metadata/calibrationInformation/'
    calibration_information_group = fid.create_group(parameters)

    frequency_str = 'frequency' + frequency
    frequency_group = calibration_information_group.create_group(frequency_str)

    calibration_slantrange_vector = metadata['calibration_range_grid']
    calibration_zero_doppler_time_vector = metadata['calibration_azimuth_grid']

    for calibration_field, description in CALIBRATION_FIELD_DICT.items():

        if calibration_field == 'nes0':
            data = np.full((calibration_zero_doppler_time_vector.size,
                            calibration_slantrange_vector.size),
                           metadata['NESZ'],
                           dtype=np.float32)
        else:
            data = np.zeros((calibration_zero_doppler_time_vector.size,
                             calibration_slantrange_vector.size),
                            dtype=np.float32)

        calibration_group = frequency_group.create_group(calibration_field)

        for pol in pol_list:

            lut_description = f'calibration {calibration_field}'

            time_units = metadata['ref_epoch_attr']

            _create_lut_coordinate_vectors(
                calibration_group,
                calibration_zero_doppler_time_vector,
                calibration_slantrange_vector,
                lut_description, time_units)

            # Update calibration LUT values
            calibration_group.create_dataset(pol, data=data)
            calibration_group[pol].attrs['description'] = np.bytes_(
                description)
            calibration_group[pol].attrs['units'] = np.bytes_('1')


def update_doppler(fid, metadata, frequency):  # time, position, velocity,
    """
    Update HDF5 file for Doppler, FM rate, and effective velocity.
    """
    # Get doppler group from metadata
    parameters = 'science/LSAR/RSLC/metadata/processingInformation/parameters'
    if parameters not in fid:
        parameters_group = fid.create_group(parameters)
    else:
        parameters_group = fid[parameters]

    calibration_slantrange_vector = metadata['calibration_range_grid']
    calibration_zero_doppler_time_vector = metadata['calibration_azimuth_grid']
    time_units = metadata['ref_epoch_attr']

    processing_information_description = 'processing information'

    _create_lut_coordinate_vectors(
            parameters_group,
            calibration_zero_doppler_time_vector,
            calibration_slantrange_vector,
            processing_information_description, time_units)

    frequency_str = 'frequency' + frequency
    doppler_group = parameters_group.create_group(frequency_str)

    if frequency == 'A':
        _create_lut_coordinate_vectors(
            doppler_group,
            calibration_zero_doppler_time_vector,
            calibration_slantrange_vector,
            processing_information_description, time_units)

    rgvals = doppler_group['slantRange'][()]
    azsecs = doppler_group['zeroDopplerTime'][()]

    if 'Doppler coeffs km' in metadata.keys():
        doppler_coeff = metadata['Doppler coeffs km']
        rgvals_in_km = rgvals / 1000.0
        dop_vals = np.polyval(doppler_coeff[::-1], rgvals_in_km)
        dop_vals = np.tile(dop_vals, (len(azsecs), 1))
    else:
        dop_coeffs = metadata['Doppler coeffs rbin']
        range_bin = np.arange(0, len(rgvals), 1.0)
        dop_vals = np.polyval(dop_coeffs[::-1], range_bin)
        dop_vals = np.tile(dop_vals, (len(azsecs), 1))

    # Update Doppler values
    doppler_group.create_dataset(
        'dopplerCentroid', data=np.asarray(dop_vals, dtype=np.float64))
    doppler_group['dopplerCentroid'].attrs['description'] = np.bytes_(
        '2D LUT of Doppler Centroid for Frequency ' + frequency)
    doppler_group['dopplerCentroid'].attrs['units'] = np.bytes_('Hz')


if __name__ == "__main__":
    '''
    Main driver.
    '''

    # Parse command line
    args = parse_args()

    # Process the data
    process(args=args)
