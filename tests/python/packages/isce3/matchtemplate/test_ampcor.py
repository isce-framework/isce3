from osgeo import gdal
import isce3
import iscetest
import numpy
import os


def create_empty_dataset(
    filename, width, length, bands, dtype, interleave="bip", file_type="ENVI"
):
    """
    Create empty dataset with user-defined options
    """
    driver = gdal.GetDriverByName(file_type)
    driver.Create(
        filename,
        xsize=width,
        ysize=length,
        bands=bands,
        eType=dtype,
        options=[f"INTERLEAVE={interleave}"],
    )


def test_ampcor():
    try:
        impls = (
            isce3.cuda.matchtemplate.PyCuAmpcor,
            isce3.matchtemplate.PyCPUAmpcor,
        )
    except AttributeError:
        # Fall back to CPU only if not compiled with CUDA support
        impls = (isce3.matchtemplate.PyCPUAmpcor,)
    for impl in impls:
        for ovs in (0, 1):  # test FFT and sinc oversamplers
            ampcor = impl()

            ampcor.useMmap = 1

            datadir = os.path.join(
                iscetest.data, "ampcor", "accuracy-testdata", "ovs128-rho0.8"
            )

            ref = os.path.join(datadir, "img1_WN_512x512_1x1_128")
            ref_raster = isce3.io.Raster(ref)
            width = ref_raster.width
            length = ref_raster.length
            ampcor.referenceImageName = ref
            ampcor.referenceImageWidth = width
            ampcor.referenceImageHeight = length

            sec = os.path.join(datadir, "img2_WN_512x512_1x1_128")
            sec_raster = isce3.io.Raster(sec)
            ampcor.secondaryImageName = sec
            assert width == sec_raster.width
            assert length == sec_raster.length
            ampcor.secondaryImageWidth = width
            ampcor.secondaryImageHeight = length

            ampcor.windowSizeWidth = 64
            ampcor.windowSizeHeight = 32
            ampcor.halfSearchRangeAcross = 20
            ampcor.halfSearchRangeDown = 20
            ampcor.skipSampleAcross = 32
            ampcor.skipSampleDown = 32

            margin = 0
            margin_rg = (
                2 * margin + 2 * ampcor.halfSearchRangeAcross + ampcor.windowSizeWidth
            )
            margin_az = (
                2 * margin + 2 * ampcor.halfSearchRangeDown + ampcor.windowSizeHeight
            )

            offset_width = (width - margin_rg) // ampcor.skipSampleAcross
            ampcor.numberWindowAcross = offset_width
            offset_length = (length - margin_az) // ampcor.skipSampleDown
            ampcor.numberWindowDown = offset_length

            ampcor.referenceStartPixelAcrossStatic = margin + ampcor.halfSearchRangeAcross
            ampcor.referenceStartPixelDownStatic = margin + ampcor.halfSearchRangeDown

            ampcor.algorithm = 0  # frequency
            ampcor.corrSurfaceOverSamplingMethod = ovs
            ampcor.derampMethod = 1

            ampcor.corrStatWindowSize = 21
            ampcor.corrSurfaceZoomInWindow = 8

            ampcor.offsetImageName = "dense_offsets"
            ampcor.grossOffsetImageName = "gross_offset"
            ampcor.snrImageName = "snr"
            ampcor.covImageName = "covariance"
            ampcor.corrImageName = "correlation_peak"

            ampcor.rawDataOversamplingFactor = 2
            ampcor.corrSurfaceOverSamplingFactor = 64

            ampcor.numberWindowAcrossInChunk = 2
            ampcor.numberWindowDownInChunk = 1

            ampcor.setupParams()
            ampcor.setConstantGrossOffset(0, 0)

            ampcor.checkPixelInImageRange()
            create_empty_dataset(
                "dense_offsets",
                ampcor.numberWindowAcross,
                ampcor.numberWindowDown,
                2,
                gdal.GDT_Float32,
            )
            create_empty_dataset(
                "gross_offsets",
                ampcor.numberWindowAcross,
                ampcor.numberWindowDown,
                2,
                gdal.GDT_Float32,
            )
            create_empty_dataset(
                "snr",
                ampcor.numberWindowAcross,
                ampcor.numberWindowDown,
                1,
                gdal.GDT_Float32,
            )
            create_empty_dataset(
                "covariance",
                ampcor.numberWindowAcross,
                ampcor.numberWindowDown,
                3,
                gdal.GDT_Float32,
            )
            create_empty_dataset(
                "correlation_peak",
                ampcor.numberWindowAcross,
                ampcor.numberWindowDown,
                1,
                gdal.GDT_Float32,
            )
            ampcor.runAmpcor()

            # Compare results to golden output
            for fname in (
                "covariance",
                "dense_offsets",
                "gross_offsets",
                "snr",
                "correlation_peak",
            ):
                print("comparing", fname)
                golden_path = os.path.join(datadir, "golden", fname)
                expected = numpy.fromfile(golden_path, dtype=numpy.float32)
                got = numpy.fromfile(fname, dtype=numpy.float32)

                assert len(got) == len(expected)

                if fname == "dense_offsets":
                    meantol = 2e-2
                    tol = 1e-1
                elif fname == "correlation_peak":
                    meantol = 1e-2
                else:
                    meantol = 1 / 64 / 5
                    tol = 1 / 64

                for i in range(len(got)):
                    if abs(got[i] - expected[i]) > tol:
                        print(
                            "got",
                            got[i],
                            "but expected",
                            expected[i],
                            "diff is",
                            abs(got[i] - expected[i]),
                        )
                        print("at index", i)
                        assert False

                meandiff = numpy.mean(abs(got - expected))
                print("meandiff", meandiff)
                assert meandiff < meantol
