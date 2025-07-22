from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from functools import cached_property
from pathlib import Path
from typing import Any, TypeVar, overload

import h5py
import numpy as np

import isce3

#: The directory within the Python package containing XML product specification
# documents for NISAR products.
XML_SPECS_DIR = Path(__file__).parent / "XML"

T = TypeVar("T")
ProductSpecT = TypeVar("ProductSpecT", bound="ProductSpec")
NumPyScalarT = TypeVar("NumPyScalarT", bound=np.generic)


def get_unique_xml_element(
    tree_or_element: ET.ElementTree | ET.Element,
    pathspec: str,
) -> ET.Element:
    """
    Find a unique XML element in a tree or sub-tree.

    Parameters
    ----------
    tree_or_element : xml.etree.ElementTree.ElementTree or xml.etree.ElementTree.Element
        The element tree or the root element of a sub-tree to search.
    pathspec : str
        The tag or path of the desired element. Matching elements are found as though by
        ``tree_or_element.findall(pathspec)``.

    Returns
    -------
    xml.etree.ElementTree.Element
        The single XML element matching `pathspec` that is a descendant of
        `tree_or_element`.

    Raises
    ------
    ValueError
        If the search query matched no elements or if it matched multiple elements.
    """
    elements = tree_or_element.findall(pathspec)

    if len(elements) < 1:
        raise ValueError(f"no xml elements found matching {pathspec!r}")
    if len(elements) > 1:
        err_msg = (
            f"multiple xml elements found matching {pathspec!r}:"
            f"{[ET.tostring(element) for element in elements]}"
        )
        raise ValueError(err_msg)

    return elements[0]


@overload
def get_xml_attrib(element: ET.Element, key: str) -> str:
    """
    Get an attribute from an XML element.

    Parameters
    ----------
    element : xml.etree.ElementTree.Element
        The XML element.
    key : str
        The attribute name.

    Returns
    -------
    str
        The value of the attribute.

    Raises
    ------
    ValueError
        If the XML element has no such attribute.
    """


@overload
def get_xml_attrib(element: ET.Element, key: str, *, type: Callable[[str], T]) -> T:
    """
    Get an attribute from an XML element.

    Parameters
    ----------
    element : xml.etree.ElementTree.Element
        The XML element.
    key : str
        The attribute name.
    type : callable
        A type converter that may be applied to convert the type of the attribute.

    Returns
    -------
    object
        The value of the attribute, with the given `type`.

    Raises
    ------
    ValueError
        If the XML element has no such attribute.
    TypeError
        If the attribute value cannot be converted to `type`.
    """


def get_xml_attrib(element, key, *, type=None):
    attrib_dict = element.attrib

    try:
        val = attrib_dict[key]
    except KeyError as e:
        errmsg = f"xml element {ET.tostring(element)} has no attribute {key!r}"
        raise ValueError(errmsg) from e

    if type is None:
        return val
    try:
        return type(val)
    except Exception as e:
        err_msg = (
            f"failed to cast attribute {key!r} of xml element {ET.tostring(element)} to"
            f" {type}"
        )
        raise TypeError(err_msg) from e


class DatasetSpec:
    """
    Specification for a Dataset in a NISAR product.

    A `DatasetSpec` is a record in a product specification XML document that represents
    a particular HDF5 Dataset in a NISAR product. It provides information about the
    expected datatype and attributes of the Dataset.

    Attributes
    ----------
    name : str
        The name of the HDF5 Dataset.
    dtype : numpy.dtype
        The datatype of the Dataset, according to the specification document.
    description : str
        The Dataset description from the specification document.
    attrs : dict
        A dict of Dataset attributes found in the specification document. Keys in the
        dict are attributes that the HDF5 Dataset is expected to contain. Values are
        string representations of the corresponding attribute value or a description of
        the attribute.
    """

    def __init__(self, element: ET.Element):
        """
        Create a new `DatasetSpec` object.

        Parameters
        ----------
        element : xml.etree.ElementTree.Element
            The XML element containing the Dataset specification.
        """
        self._element = element

    @cached_property
    def name(self) -> str:
        return get_xml_attrib(self._element, "name")

    def _is_signed(self) -> bool:
        """Check if the Dataset is signed or unsigned (integer-valued Datasets only)."""
        val = get_xml_attrib(self._element, "signed")

        if val == "true":
            return True
        if val == "false":
            return False

        raise ValueError(
            f"malformed xml spec: 'signed' attribute of element"
            f" {ET.tostring(self._element)} has unexpected value {val!r}; expected"
            " 'true' or 'false'"
        )

    def _is_complex(self) -> bool:
        """Check if the Dataset is complex-valued."""
        # For complex-valued datasets, there should be a single child element with the
        # tag `annotation` and `app='io'` and `kwd='complex'` attributes. Other datasets
        # should have no such child element.
        pathspec = "./annotation[@app='io'][@kwd='complex']"
        elements = self._element.findall(pathspec)

        if len(elements) > 1:
            raise ValueError(
                f"malformed xml spec: element {ET.tostring(self._element)} contains"
                " multiple child elements indicating that it is complex-valued"
            )

        return len(elements) == 1

    @cached_property
    def dtype(self) -> np.dtype:
        tag = self._element.tag

        # String datasets should be stored in the HDF5 file as byte strings
        # (`numpy.bytes_` objects), since HDF5 has relatively poor support for Unicode
        # strings. `length='0'` in the spec indicates that the string should be
        # dynamically-sized to fit the Dataset contents.
        if tag == "string":
            length = get_xml_attrib(self._element, "length", type=int)
            return np.dtype("S" if (length == 0) else f"S{length}")

        elif tag == "integer":
            is_signed = self._is_signed()
            width = get_xml_attrib(self._element, "width", type=int)
            return np.dtype(f"{'int' if is_signed else 'uint'}{width}")

        # XXX: The 'real' tag is used for both real- and complex-valued floating-point
        # Datasets.
        elif tag == "real":
            is_complex = self._is_complex()
            width = get_xml_attrib(self._element, "width", type=int)

            if is_complex:
                if width == 16:
                    # NumPy currently doesn't support a 'complex32' dtype so return a
                    # custom dtype in this case.
                    return isce3.core.types.complex32
                else:
                    # If the dataset is complex-valued, the width stored in the XML spec
                    # is the width of an individual component. The width of each
                    # (real,imag) pair is twice the width of an individual component.
                    # The NumPy convention is to name complex datatypes using the width
                    # of the entire datum, including both the real and imag components.
                    return np.dtype(f"complex{2 * width}")
            else:
                return np.dtype(f"float{width}")

        raise ValueError(
            f"malformed xml spec: the tag of element {ET.tostring(self._element)} is"
            f" {tag!r}; expected 'string', 'integer', or 'real'"
        )

    def _get_attrs_child_element(self) -> ET.Element:
        """
        Get the child XML element that stores the Dataset's attributes and description.
        """
        # Get the child element that contains the attributes to be populated by the
        # corresponding HDF5 dataset. It should always have the tag `annotation` and
        # `app='conformance'` and `lang='en'` attributes.
        pathspec = "./annotation[@app='conformance'][@lang='en']"
        return get_unique_xml_element(self._element, pathspec)

    @cached_property
    def description(self) -> str:
        return self._get_attrs_child_element().text

    @cached_property
    def attrs(self) -> dict[str, str]:
        attrs_element = self._get_attrs_child_element()
        return {
            key: val
            for (key, val) in attrs_element.items()
            if key not in {"app", "conformance", "lang"}
        }


class ProductSpec:
    """
    NISAR product specification.

    A `ProductSpec` can be used to access information about the structure and contents
    of a NISAR product from the product's specification XML document.

    Attributes
    ----------
    global_attrs : dict
        A dict of global attributes found in the specification document. The contents of
        the dict are attributes that the root HDF5 Group is expected to contain.
    """

    def __init__(self, tree: ET.ElementTree):
        """
        Create a new `ProductSpec` object.

        Parameters
        ----------
        element : xml.etree.ElementTree.ElementTree
            The XML element tree containing the product specification.
        """
        self._tree = tree

    @classmethod
    def from_file(cls: type[ProductSpecT], xml_path: str | os.PathLike) -> ProductSpecT:
        """
        Create a new `ProductSpec` from an XML file.

        Parameters
        ----------
        xml_path : path-like
            The path to the XML file.
        """
        tree = ET.parse(xml_path)
        return cls(tree)

    @cached_property
    def global_attrs(self) -> dict[str, str]:
        element = get_unique_xml_element(self._tree, "./product/science/annotation")
        return {key: val for (key, val) in element.items() if key != "app"}

    def get_dataset_spec(self, name: str) -> DatasetSpec:
        """
        Get the specification for a Dataset in the product.

        Parameters
        ----------
        name : str
            The name of the Dataset as it appears in the XML document.

        Returns
        -------
        DatasetSpec
            The Dataset specification.
        """
        pathspec = f"./product/science/nodes/*[@name={name!r}]"
        element = get_unique_xml_element(self._tree, pathspec)
        return DatasetSpec(element)

    def iter_dataset_specs(self) -> Iterator[DatasetSpec]:
        """
        Iterate over specifications for all Datasets in the product.

        Yields
        ------
        DatasetSpec
            The specification for a Dataset in the product.
        """
        element = get_unique_xml_element(self._tree, "./product/science/nodes")
        for subelement in element:
            yield DatasetSpec(subelement)


def get_product_spec(product_type: str) -> ProductSpec:
    """
    Get the product specification for a given NISAR product.

    Parameters
    ----------
    product_type : {'GCOV', 'GSLC'}
        The NISAR product type. Only 'GCOV' and 'GSLC' are currently supported.

    Returns
    -------
    ProductSpec
        The specification for the specified product type.
    """
    product_type = product_type.upper()

    if product_type in {"GCOV", "GSLC"}:
        xml_path = XML_SPECS_DIR / f"L2/nisar_L2_{product_type}.xml"
    elif product_type in {"GOFF", "GUNW", "RIFG", "ROFF", "RSLC", "RUNW", "STATIC"}:
        raise NotImplementedError(f"unsupported product type {product_type!r}")
    else:
        raise ValueError(f"unexpected product type {product_type!r}")

    return ProductSpec.from_file(xml_path)


def populate_global_attrs_from_spec(
    hdf5_file: h5py.File, product_spec: ProductSpec
) -> None:
    """
    Set NISAR product global attributes from the product specification.

    Updates the `attrs` dict of the root Group of a NISAR product HDF5 file with values
    obtained from the product's specification XML document.

    All global attributes are expected to be strings and are stored as byte strings with
    UTF-8 encoding.

    Parameters
    ----------
    hdf5_file : h5py.File
        The HDF5 file.
    product_spec : ProductSpec
        Metadata obtained from the product specification XML document.
    """
    for key, val in product_spec.global_attrs.items():
        # All global attributes are expected to be strings.
        if not isinstance(val, str):
            raise TypeError(
                f"{key!r} attribute value {val} has type {type(val)}, expected str"
            )
        hdf5_file.attrs[key] = np.bytes_(val)


def populate_dataset_attrs_from_spec(
    hdf5_dataset: h5py.Dataset, dataset_spec: DatasetSpec
) -> None:
    """
    Set HDF5 Dataset attributes from the NISAR product specification.

    Updates the `attrs` dict of an HDF5 Dataset in a NISAR product with values obtained
    from the product's specification XML document. The following attributes will be
    added if found in the product specification:

    - description
    - grid_mapping
    - license
    - long_name
    - standard_name
    - valid_max
    - valid_min
    - _FillValue

    Other attributes such as `mean_value`, `epsg_code`, and `units`, which may depend on
    parameters determined at runtime by the product generation workflow, are not
    populated. Additional attributes such as `DIMENSION_LIST` and `REFERENCE_LIST`,
    which may be automatically populated by the HDF5 library, are also ignored.

    String-valued attributes are stored as byte strings with UTF-8 encoding. Attributes
    that represent numeric quantities in the intrinsic units of the Dataset (e.g.
    `valid_min`) are converted to the same datatype as the Dataset.

    Parameters
    ----------
    hdf5_dataset : h5py.Dataset
        The HDF5 Dataset.
    dataset_spec : DatasetSpec
        Dataset metadata obtained from the product specification XML document.
    """
    # Set the HDF5 Dataset 'description' attribute.
    hdf5_dataset.attrs["description"] = np.bytes_(dataset_spec.description)

    # If `key` is found in the XML spec attributes, convert its value to `type` and add
    # it to the HDF5 Dataset attributes. Otherwise, do nothing.
    def soft_copy_attr(key: str, type: Callable[[str], Any]) -> None:
        try:
            val = dataset_spec.attrs[key]
        except KeyError:
            pass
        else:
            hdf5_dataset.attrs[key] = type(val)

    # Set other string-valued attributes whose values can be copied from the XML.
    for key in ["grid_mapping", "license", "long_name", "standard_name"]:
        soft_copy_attr(key, np.bytes_)

    # Get the NumPy scalar type corresponding to the Dataset's datatype.
    scalar_type = hdf5_dataset.dtype.type

    # Set any numeric attributes (except '_FillValue' -- see below) that can be copied
    # from the XML.
    for key in ["valid_min", "valid_max"]:
        soft_copy_attr(key, scalar_type)

    # Most fill value strings can be directly converted to the corresponding NumPy
    # scalar type, but it doesn't work for '(nan+nan*j)', so we need to handle that case
    # specially.
    def convert_fill_value(fill_value_str: str):
        if fill_value_str == "(nan+nan*j)":
            return scalar_type(np.nan + np.nan * 1j)
        else:
            return scalar_type(fill_value_str)

    # Set the '_FillValue' attribute if found in the XML.
    soft_copy_attr("_FillValue", convert_fill_value)
