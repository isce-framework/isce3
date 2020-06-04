# goofy but portable

runconfig = """
# Structure inherited from L0 PGEs covers inputs, outputs, and identification.
runconfig:
    groups:
        InputFileGroup:
            # REQUIRED List of NISAR raw data products in L0B format.
            InputFilePath:
            - /tmp/ALPSRP037370690.L0B.h5

        DynamicAncillaryFileGroup:
            # Digital elevation model, optional.
            DEMFile: null

            # Refined orbit, optional.
            # Defaults to orbit within L0B product.
            Orbit: null

            # Refined pointing, optional.
            # Defaults to attitude within L0B product.
            Pointing: null

            # External calibration data, optional.
            # Defaults to no extra calibration gain, phase, delay, etc.
            ExternalCalibration: null

            # Internal calibration tables, optional.
            # If not provided, no secondary elevation antenna pattern correction.
            InternalCalibration: null

            # Polarimetric calibration data, optional.  Crosstalk, relative phases, etc.
            # If not provided, no polarimetric calibration performed.
            PolarimetricCalibration: null

            # Pre- and post-data take calibration data products, optional.
            BookendCalibration: null

            # Antenna pattern data, optional.
            # Defaults to sinc^4 pattern using nominal antenna dimensions.
            AntennaPattern: null

            # Chirp replica file, optional.
            # If absent will generate LFM chirp using parameters in L0B product.
            Waveform: null

        ProductPathGroup:
            # Directory where PGE will place results. Irrelevant to SAS.
            ProductPath: /out
            # Directory where SAS can write temporary data
            ScratchPath: .
            # SAS writes output product to the following file. PGE may rename.
            # NOTE: R2 needs to handle mixed-mode case with multiple outputs.
            SASOutputFile: ./slc.h5

        PrimaryExecutable:
            # Handy to set to "SLC" until other ISCE tools updated.
            ProductType: RSLC

        Geometry:
            RelativeOrbitNumber: 1
            FrameNumber: 1

# TODO
archive: {}

# TODO
qaqc: {}

# Whether or not to use GPU, optional. Defaults to True if available.
worker:
    gpu_enabled: True

# TODO
profiler: {}

processing:
    output_grid:
        # Azimuth bounds of output SLC, optional.
        # Defaults to extent of raw data minus sythetic aperture and shifted
        # by squint.
        start_time: null
        end_time: null

        # Range bounds of output SLC in meters, optional.
        # Defaults to entire fully focused swath.
        start_range: null
        end_range: null

        # Output grid azimuth sample rate in Hz, optional.
        # Defaults to input PRF.
        output_prf: null

        # Interval for azimuth grid alignment in seconds.  See docs.
        # Typically want 1 second and integer PRF (in Hz).  Will not override
        # user-provided start_time.
        time_snap_interval: 1.0

    # Range spectral window, optional.  Defaults to no weighting.
    range_window:
        # Kaiser or Cosine
        kind: Kaiser
        # Shape parameter. For Kaiser, 0 <= shape < Inf.  For Cosine, 0 <= shape <= 1
        shape: 0.0

    # Azimuth spectral window, optional.  Defaults to no weighting.
    azimuth_window:
        kind: Kaiser
        shape: 0.0

    # Range filter parameters for mixed-mode cases.
    range_common_band_filter:
        # Stop-band attenuation in dB
        attenuation: 40.0
        # Transition width as a fraction of output bandwidth.
        width: 0.15

    doppler:
        # Offset between quaternion frame and antenna boresight in degrees.
        # TBD This will likely be parameter in a separate cal file.
        azimuth_boresight_deg: 0.9

        # How to interpolate between samples in the LUT.
        interp_method: bilinear

        # Postings for generated Doppler lookup table.
        spacing:
            # Lookup table range spacing in m
            range: 2000.0
            # Lookup table Azimuth spacing in s
            azimuth: 1.0

        rdr2geo:
            # Slant range convergence threshold (m): float (default: 1e-8)
            threshold: 1e-8
            # Max number of primary iterations: int (default: 25)
            maxiter: 25
            # Max number of secondary iterations: int (default: 15)
            extraiter: 15

    # Settings for range compression algorithm.
    rangecomp:
        # Convolution output mode: {"valid", "full", "same"} (default: "full")
        mode: full

        # Range compression will always process the full pulse, so the range
        # dimension will be ignored.
        block_size:
            range: 0
            azimuth: 1024

    # Settings for azimuth compression algorithm.
    azcomp:
        # Azimuth compression can be tiled arbitrarily, though dimensions will
        # affect runtime.
        block_size:
            range: 64
            azimuth: 64

        # Desired azimuth resolution in meters.
        azimuth_resolution: 6.0

        kernel:
            # Knab or NFFT
            type: Knab
            # Length = 1+2*halfWidth
            halfwidth: 4
            # Transform padding ratio for NFFT method.
            approx_oversample: 1.7

            fit: Table # null or Cheby or Table
            fit_order: 2048

        rdr2geo:
            # Slant range convergence threshold (m): float (default: 1e-8)
            threshold: 1e-8
            # Max number of primary iterations: int (default: 25)
            maxiter: 25
            # Max number of secondary iterations: int (default: 15)
            extraiter: 15

        geo2rdr:
            # Slant range convergence threshold (m): float (default: 1e-8)
            threshold: 1e-8
            # Max number of iterations: int (default: 50)
            maxiter: 50
            # Step size for computing numerical gradient of Doppler (m): float
            # (default: 10.)
            delta_range: 10.

    dry_troposphere_model: tsx

    dem:
        # Height (in meters) to use if DEM unavailable.
        reference_height: 0.0

        # How to interpolate the digital elevation model.  One of
        # nearest, bilinear, bicubic, biquintic, or sinc
        interp_method: biquintic

    # Nominal antenna dimensions to use for BLU, EAP, etc. when no antenna
    # pattern input file is provided or its contents are unsuitable.
    # Each dimension has units of meters and is assumed 12.0 m if unspecified.
    nominal_antenna_size:
        range: 12.0
        azimuth: 12.0

    # Scale factor to apply to data before float16 encoding, optional.
    # Default is 1.0.
    # The largest representable float16 value is 65504.
    # NOTE This is ad-hoc until the processor is radiometrically calibrated.
    encoding_scale_factor: 1.0

    # Processing stage switches, mostly for debug.
    # Any missing stages assumed True
    is_enabled:
        # radio frequency interference mitigation
        rfi_removal: True
        # azimuth resampling and gap-filling
        presum_blu: True
        # range compression
        rangecomp: True
        # elevation antenna pattern correction
        eap: True
        # R^4 spreading loss correction
        range_cor: True
        # azimuth compression
        azcomp: True
"""
