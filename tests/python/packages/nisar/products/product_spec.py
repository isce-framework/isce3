import textwrap
import xml.etree.ElementTree as ET
from collections.abc import Generator
from contextlib import contextmanager
from tempfile import NamedTemporaryFile

import h5py
import nisar.products
import numpy as np
import pytest

import isce3
import nisar
from nisar.products import (
    ProductSpec,
    populate_global_attrs_from_spec,
    populate_dataset_attrs_from_spec,
)


@pytest.fixture
def gcov_product_spec() -> ProductSpec:
    return nisar.products.get_product_spec("GCOV")


@pytest.fixture
def partial_gcov_product_spec() -> ProductSpec:
    xml = textwrap.dedent(
        """
        <?xml version="1.0"?>
        <algorithm name="L2_GeocodedCovariance"
                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
          <product template="L2_GCOV_template">
            <science uid="sci_1">
              <nodes>
                <integer name="/science/LSAR/identification/absoluteOrbitNumber"
                         shape="scalar"
                         signed="false"
                         width="32">
                  <annotation app="conformance"
                              lang="en">Absolute orbit number</annotation>
                </integer>
                <integer name="/science/LSAR/identification/trackNumber"
                         shape="scalar"
                         signed="false"
                         width="32">
                  <annotation app="conformance"
                              lang="en">Track number</annotation>
                </integer>
                <integer name="/science/LSAR/identification/frameNumber"
                         shape="scalar"
                         signed="false"
                         width="16">
                  <annotation app="conformance"
                              lang="en">Frame number</annotation>
                </integer>
              </nodes>
            </science>
          </product>
        </algorithm>
        """.strip()
    )
    element = ET.fromstring(xml)
    tree = ET.ElementTree(element)
    return ProductSpec(tree)


class TestProductSpec:
    def test_get_global_attrs(self, gcov_product_spec: ProductSpec):
        assert gcov_product_spec.global_attrs == dict(
            Conventions="CF-1.7",
            title="NISAR L2 GCOV Product",
            institution="NASA JPL",
            mission_name="NISAR",
            reference_document=(
                "D-102274 NISAR NASA SDS Product Specification Level-2 Geocoded"
                " Polarimetric Covariance L2 GCOV"
            ),
            contact="nisar-sds-ops@jpl.nasa.gov",
        )

    def test_get_dataset_spec(self, gcov_product_spec: ProductSpec):
        name = "/science/LSAR/identification/absoluteOrbitNumber"
        dataset_spec = gcov_product_spec.get_dataset_spec(name)
        assert dataset_spec.name == name

    def test_iter_dataset_specs(self, partial_gcov_product_spec: ProductSpec):
        dset_specs = list(partial_gcov_product_spec.iter_dataset_specs())
        assert len(dset_specs) == 3
        assert dset_specs[0].name == "/science/LSAR/identification/absoluteOrbitNumber"
        assert dset_specs[1].name == "/science/LSAR/identification/trackNumber"
        assert dset_specs[2].name == "/science/LSAR/identification/frameNumber"

    def test_missing_dataset_spec(self, gcov_product_spec: ProductSpec):
        name = "/science/LSAR/identification/asdf"
        with pytest.raises(ValueError, match="^no xml elements found matching"):
            gcov_product_spec.get_dataset_spec(name)

    def test_duplicate_dataset_specs(self):
        xml = textwrap.dedent(
            """
            <?xml version="1.0"?>
            <algorithm name="L2_GeocodedCovariance"
                       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
              <product template="L2_GCOV_template">
                <science uid="sci_1">
                  <nodes>
                    <integer name="/science/LSAR/identification/absoluteOrbitNumber"
                             shape="scalar"
                             signed="false"
                             width="32">
                      <annotation app="conformance"
                                  lang="en">Absolute orbit number</annotation>
                    </integer>
                    <integer name="/science/LSAR/identification/absoluteOrbitNumber"
                             shape="scalar"
                             signed="false"
                             width="32">
                      <annotation app="conformance"
                                  lang="en">Absolute orbit number</annotation>
                    </integer>
                  </nodes>
                </science>
              </product>
            </algorithm>
            """.strip()
        )
        element = ET.fromstring(xml)
        tree = ET.ElementTree(element)
        product_spec = ProductSpec(tree)

        name = "/science/LSAR/identification/absoluteOrbitNumber"
        with pytest.raises(ValueError, match="^multiple xml elements found matching"):
            product_spec.get_dataset_spec(name)


class TestDatasetSpec:
    def test_frame_number(self, gcov_product_spec: ProductSpec):
        name = "/science/LSAR/identification/frameNumber"
        dataset_spec = gcov_product_spec.get_dataset_spec(name)

        assert dataset_spec.name == name
        assert dataset_spec.dtype == np.uint16
        assert dataset_spec.description == "Frame number"
        assert dataset_spec.attrs == {}

    def test_mission_id(self, gcov_product_spec: ProductSpec):
        name = "/science/LSAR/identification/missionId"
        dataset_spec = gcov_product_spec.get_dataset_spec(name)

        assert dataset_spec.name == name
        assert dataset_spec.dtype == np.dtype(np.bytes_)
        assert dataset_spec.description == "Mission identifier"
        assert dataset_spec.attrs == {}

    def test_zero_doppler_start_time(self, gcov_product_spec: ProductSpec):
        name = "/science/LSAR/identification/zeroDopplerStartTime"
        dataset_spec = gcov_product_spec.get_dataset_spec(name)

        assert dataset_spec.name == name
        assert dataset_spec.dtype == np.dtype("S29")
        assert dataset_spec.description == (
            "Azimuth start time (in UTC) of the product in the format"
            " YYYY-mm-ddTHH:MM:SS.sssssssss"
        )
        assert dataset_spec.attrs == {}

    def test_hhhh(self, gcov_product_spec: ProductSpec):
        name = "/science/LSAR/GCOV/grids/frequencyA/HHHH"
        dataset_spec = gcov_product_spec.get_dataset_spec(name)

        assert dataset_spec.name == name
        assert dataset_spec.dtype == np.float32
        assert dataset_spec.description == "Covariance between HH and HH"
        assert dataset_spec.attrs == dict(
            long_name="Geocoded polarimetric covariance term HHHH",
            mean_value="Arithmetic average of the numeric data points",
            min_value="Minimum value of the numeric data points",
            max_value="Maximum value of the numeric data points",
            sample_stddev="Standard deviation of the numeric data points",
            valid_min="0",
            DIMENSION_LIST="HDF5 internal attribute",
            _FillValue="nan",
            grid_mapping="projection",
            units="1",
        )

    def test_hhvv(self, gcov_product_spec: ProductSpec):
        name = "/science/LSAR/GCOV/grids/frequencyA/HHVV"
        dataset_spec = gcov_product_spec.get_dataset_spec(name)

        # This string with placeholders is helpful for reducing copy & paste in the
        # descriptions of various attributes below.
        template = "{} of the {} part of the numeric data points"

        assert dataset_spec.name == name
        assert dataset_spec.dtype == np.complex64
        assert dataset_spec.description == "Covariance between HH and VV"
        assert dataset_spec.attrs == dict(
            DIMENSION_LIST="HDF5 internal attribute",
            long_name="Geocoded polarimetric covariance term HHVV",
            mean_real_value=template.format("Arithmetic average", "real"),
            min_real_value=template.format("Minimum value", "real"),
            max_real_value=template.format("Maximum value", "real"),
            sample_stddev_real=template.format("Standard deviation", "real"),
            mean_imag_value=template.format("Arithmetic average", "imaginary"),
            min_imag_value=template.format("Minimum value", "imaginary"),
            max_imag_value=template.format("Maximum value", "imaginary"),
            sample_stddev_imag=template.format("Standard deviation", "imaginary"),
            _FillValue="(nan+nan*j)",
            grid_mapping="projection",
            units="1",
        )

    def test_projection(self, gcov_product_spec: ProductSpec):
        name = "/science/LSAR/GCOV/grids/frequencyA/projection"
        dataset_spec = gcov_product_spec.get_dataset_spec(name)

        assert dataset_spec.name == name
        assert dataset_spec.dtype == np.uint32
        assert dataset_spec.description == (
            "Product map grid projection: EPSG code, with additional projection"
            " information as HDF5 Attributes"
        )
        assert dataset_spec.attrs == dict(
            ellipsoid="Projection ellipsoid",
            epsg_code="Projection EPSG code",
            false_easting=(
                "The value added to all abscissa values in the rectangular coordinates"
                " for a map projection."
            ),
            false_northing=(
                "The value added to all ordinate values in the rectangular coordinates"
                " for a map projection."
            ),
            grid_mapping_name="Grid mapping variable name",
            inverse_flattening="Inverse flattening of the ellipsoidal figure",
            latitude_of_projection_origin=(
                "The latitude chosen as the origin of rectangular coordinates for a map"
                " projection."
            ),
            longitude_of_projection_origin=(
                "The longitude, with respect to Greenwich, of the prime meridian"
                " associated with the geodetic datum."
            ),
            longitude_of_central_meridian=(
                "The line of longitude at the center of a map projection generally used"
                " as the basis for constructing the projection."
            ),
            scale_factor_at_central_meridian=(
                "A multiplier for reducing a distance obtained from a map by"
                " computation or scaling to the actual distance along the central meridian."
            ),
            semi_major_axis="Semi-major axis",
            spatial_ref="Spatial reference",
            utm_zone_number="UTM zone number",
        )

    def test_complex32(self):
        xml = textwrap.dedent(
            """
            <?xml version="1.0"?>
            <real name="/science/LSAR/RSLC/swaths/frequencyA/HH"
                  shape="complexDataFrequencyAShape"
                  width="16">
              <annotation app="conformance"
                          lang="en"
                          units="1">Focused RSLC image (HH)</annotation>
              <annotation app="io" kwd="complex" />
            </real>
            """.strip()
        )
        element = ET.fromstring(xml)
        dataset_spec = nisar.products.DatasetSpec(element)

        assert dataset_spec.name == "/science/LSAR/RSLC/swaths/frequencyA/HH"
        assert dataset_spec.dtype == isce3.core.types.complex32
        assert dataset_spec.description == "Focused RSLC image (HH)"
        assert dataset_spec.attrs == dict(units="1")

    def test_bad_signed_attr(self):
        xml = textwrap.dedent(
            """
            <?xml version="1.0"?>
            <integer name="/science/LSAR/identification/absoluteOrbitNumber"
                 shape="scalar"
                 signed="asdf"
                 width="32">
            <annotation app="conformance"
                        lang="en">Absolute orbit number</annotation>
            </integer>
            """.strip()
        )
        element = ET.fromstring(xml)
        dataset_spec = nisar.products.DatasetSpec(element)

        regex = (
            "^malformed xml spec: 'signed' attribute of element .+ has unexpected value"
            " 'asdf'; expected 'true' or 'false'$"
        )
        with pytest.raises(ValueError, match=regex):
            dataset_spec.dtype

    def test_bad_complex_dtype(self):
        xml = textwrap.dedent(
            """
            <?xml version="1.0"?>
            <real name="/science/LSAR/GCOV/grids/frequencyA/HHHV"
                  shape="complexDataFrequencyAShape"
                  width="32">
              <annotation app="conformance"
                          lang="en">Covariance between HH and HV</annotation>
              <annotation app="io" kwd="complex" />
              <annotation app="io" kwd="complex" />
            </real>
            """.strip()
        )
        element = ET.fromstring(xml)
        dataset_spec = nisar.products.DatasetSpec(element)

        regex = (
            "^malformed xml spec: element .+ contains multiple child elements"
            " indicating that it is complex-valued$"
        )
        with pytest.raises(ValueError, match=regex):
            dataset_spec.dtype

    def test_bad_dtype_tag(self):
        xml = textwrap.dedent(
            """
            <?xml version="1.0"?>
            <asdf name="/science/LSAR/identification/absoluteOrbitNumber"
                 shape="scalar"
                 signed="asdf"
                 width="32">
            <annotation app="conformance"
                        lang="en">Absolute orbit number</annotation>
            </asdf>
            """.strip()
        )
        element = ET.fromstring(xml)
        dataset_spec = nisar.products.DatasetSpec(element)

        regex = (
            "^malformed xml spec: the tag of element .+ is 'asdf'; expected 'string',"
            " 'integer', or 'real'$"
        )
        with pytest.raises(ValueError, match=regex):
            dataset_spec.dtype

    def test_missing_length_attr(self):
        xml = textwrap.dedent(
            """
            <?xml version="1.0"?>
            <string name="/science/LSAR/identification/missionId"
                    shape="scalar">
              <annotation app="conformance"
                          lang="en">Mission identifier</annotation>
            </string>
            """.strip()
        )
        element = ET.fromstring(xml)
        dataset_spec = nisar.products.DatasetSpec(element)

        regex = "^xml element .+ has no attribute 'length'$"
        with pytest.raises(ValueError, match=regex):
            dataset_spec.dtype

    def test_invalid_length_attr(self):
        xml = textwrap.dedent(
            """
            <?xml version="1.0"?>
            <string length="asdf"
                    name="/science/LSAR/identification/missionId"
                    shape="scalar">
              <annotation app="conformance"
                          lang="en">Mission identifier</annotation>
            </string>
            """.strip()
        )
        element = ET.fromstring(xml)
        dataset_spec = nisar.products.DatasetSpec(element)

        regex = "^failed to cast attribute 'length' of xml element .+ to <class 'int'>$"
        with pytest.raises(TypeError, match=regex):
            dataset_spec.dtype


class TestGetProductSpec:
    @pytest.mark.parametrize("product_type", ["GCOV", "GSLC"])
    def test_get_product_spec(self, product_type: str):
        product_spec = nisar.products.get_product_spec(product_type)
        title = product_spec.global_attrs["title"]
        assert title == f"NISAR L2 {product_type} Product"

    def test_unsupported_product_type(self):
        regex = "^unsupported product type 'RSLC'$"
        with pytest.raises(NotImplementedError, match=regex):
            nisar.products.get_product_spec("RSLC")

    def test_bad_product_type(self):
        regex = "^unexpected product type 'ASDF'$"
        with pytest.raises(ValueError, match=regex):
            nisar.products.get_product_spec("ASDF")


@contextmanager
def temporary_hdf5_file() -> Generator[h5py.File, None, None]:
    """
    Create and open a temporary HDF5 file for writing.

    The file is automatically removed from the file system upon exiting the context
    manager.

    Yields
    ------
    h5py.File
        The open HDF5 file.
    """
    with NamedTemporaryFile(suffix=".h5") as tmp_file:
        with h5py.File(tmp_file.name, "w") as hdf5_file:
            yield hdf5_file


def test_populate_global_attrs_from_spec(gcov_product_spec: ProductSpec):
    with temporary_hdf5_file() as hdf5_file:
        populate_global_attrs_from_spec(hdf5_file, gcov_product_spec)
        assert dict(hdf5_file.attrs) == dict(
            Conventions=np.bytes_("CF-1.7"),
            title=np.bytes_("NISAR L2 GCOV Product"),
            institution=np.bytes_("NASA JPL"),
            mission_name=np.bytes_("NISAR"),
            reference_document=np.bytes_(
                "D-102274 NISAR NASA SDS Product Specification Level-2 Geocoded"
                " Polarimetric Covariance L2 GCOV"
            ),
            contact=np.bytes_("nisar-sds-ops@jpl.nasa.gov"),
        )


class TestPopulateDatasetAttrsFromSpec:
    def test_number_of_looks(self, gcov_product_spec: ProductSpec):
        name = "/science/LSAR/GCOV/grids/frequencyA/numberOfLooks"
        dataset_spec = gcov_product_spec.get_dataset_spec(name)

        with temporary_hdf5_file() as hdf5_file:
            # Create a scalar Dataset and update its attributes.
            dataset = hdf5_file.create_dataset(name, shape=(), dtype=dataset_spec.dtype)
            populate_dataset_attrs_from_spec(dataset, dataset_spec)

            # Get the dict of attributes written to the Dataset.
            attrs_dict = dict(dataset.attrs)

            # Pop '_FillValue' from the dict before checking its contents since NaN !=
            # NaN. We'll check it separately below.
            fill_value = attrs_dict.pop("_FillValue")
            assert attrs_dict == dict(
                long_name=np.bytes_("Number of radar looks"),
                valid_min=np.float32(0.0),
                grid_mapping=np.bytes_("projection"),
                description=np.bytes_(
                    "Number of averaged radar-grid pixels for covariance estimation"
                ),
            )

            # Check the '_FillValue' attribute.
            assert np.isnan(fill_value)
            assert fill_value.dtype == np.float32

    def test_elevation_antenna_pattern(self, gcov_product_spec: ProductSpec):
        name = (
            "/science/LSAR/GCOV/metadata/calibrationInformation/frequencyA"
            "/elevationAntennaPattern/HH"
        )
        dataset_spec = gcov_product_spec.get_dataset_spec(name)

        with temporary_hdf5_file() as hdf5_file:
            # Create a scalar Dataset and update its attributes.
            dataset = hdf5_file.create_dataset(name, shape=(), dtype=dataset_spec.dtype)
            populate_dataset_attrs_from_spec(dataset, dataset_spec)

            # Get the dict of attributes written to the Dataset.
            attrs_dict = dict(dataset.attrs)

            # Pop '_FillValue' from the dict before checking its contents since NaN !=
            # NaN. We'll check it separately below.
            fill_value = attrs_dict.pop("_FillValue")
            assert attrs_dict == dict(
                grid_mapping=np.bytes_("projection"),
                description=np.bytes_("Complex two-way elevation antenna pattern"),
            )

            # Check the '_FillValue' attribute.
            assert np.isnan(fill_value.real)
            assert np.isnan(fill_value.imag)
            assert fill_value.dtype == np.complex64
