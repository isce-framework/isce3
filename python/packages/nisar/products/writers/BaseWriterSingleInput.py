import h5py
import warnings
import numpy as np
from datetime import datetime
import xml.etree.ElementTree as ET
import journal

from nisar.products.readers import open_product
from nisar.h5 import cp_h5_meta_data

DATE_TIME_METADATA_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'


# The units below have been tested through UDUNITS-2.
# If a unit exists in the product but is not yet included in
# the dictionary below, the writer will issue a WARNING.
# This is done to make sure that all units are "valid"
# according to UDUNITS-2. To remove the WARNING, please
# update the dictionary below
cf_units_dict = {
    '1': '1',
    'DN': '1',
    'unitless': '1',
    'seconds': 'seconds',
    'meters': 'meters',
    'degrees': 'degrees',
    'hertz': 'hertz',
    'radians': 'radians',
    'radians^-1': 'radians^-1',
    'meters / second': 'meters / second',
    'meters per second': 'meters / second',
    'meters / second ^ 2': 'meters / second ^ 2',
    'meters per second squared': 'meters / second ^ 2',
    'radians per second': 'radians / second'
}


def check_h5_dtype_vs_xml_spec(xml_metadata_entry, h5_dataset_obj,
                               verbose=False):
    """
    Check the data type of an HDF5 Dataset against the product specification
    XML

    Parameters
    ----------
    xml_metadata_entry: xml.etree.ElementTree
        ElementTree object parsed from the product specifications XML file
    h5_dataset_obj: h5py.Dataset
        Product h5py dataset
    verbose: bool
        Flag indicating if the check should be applied in verbose
        mode
    """
    warning_channel = journal.warning('check_h5_dtype_vs_xml_spec')

    full_h5_ds_path = xml_metadata_entry.attrib['name']

    xml_dtype = xml_metadata_entry.tag

    # Get length and width attributes if they exist. Otherwise, set them to
    # None.
    xml_length, xml_width = [xml_metadata_entry.attrib[attr]
                             if attr in xml_metadata_entry.attrib else None
                             for attr in ["length", "width"]]

    hdf5_dtype = h5_dataset_obj.dtype
    hdf5_dtype_is_int = 'int' in str(hdf5_dtype)
    hdf5_dtype_is_float = 'float' in str(hdf5_dtype)
    hdf5_dtype_is_complex = 'complex' in str(hdf5_dtype)
    hdf5_dtype_is_numeric = (hdf5_dtype_is_int or
                             hdf5_dtype_is_float or
                             hdf5_dtype_is_complex)

    # Invalid or unsupported data types
    if xml_dtype not in ['string', 'integer', 'real']:
        warning_channel.log(f'Unsupported data type "{xml_dtype}"'
                            ' found in the metadata field'
                            f' "{full_h5_ds_path}"')
        return

    # verify string data types (`S`: fixed-length strings)
    if (xml_dtype == 'string' and hdf5_dtype.kind != 'S'):
        warning_channel.log(f'The metadata field {full_h5_ds_path}'
                            ' is expected to be a string, but it has'
                            f' data type "{hdf5_dtype}"')

    # verify integer data types
    elif xml_dtype == 'integer' and not hdf5_dtype_is_int:
        warning_channel.log(f'The metadata field {full_h5_ds_path}'
                            ' is expected to be an integer, but it has'
                            f' data type "{hdf5_dtype}"')

    # verify real or complex data types
    elif (xml_dtype == 'real' and not hdf5_dtype_is_float and
            not hdf5_dtype_is_complex):
        warning_channel.log(f'The metadata field {full_h5_ds_path}'
                            ' is expected to be a real number, but it has'
                            f' data type "{hdf5_dtype}"')

    # verify if numeric data types contains the attribute "width"
    if hdf5_dtype_is_numeric and xml_width is None:
        warning_channel.log(f'The metadata field {full_h5_ds_path}'
                            f' has data type "{hdf5_dtype}" but the attribute'
                            ' "width" of the associated XML entry is not set')

    # verify the width of real (non-complex) values
    elif (hdf5_dtype_is_numeric and not hdf5_dtype_is_complex and
            hdf5_dtype.itemsize != int(xml_width)/8):
        warning_channel.log(f'The metadata field {full_h5_ds_path}'
                            f' has data type "{hdf5_dtype}" but the width of'
                            ' the corresponding XML entry is set to'
                            f' "{xml_width}"')

    # verify the width of complex values
    elif (hdf5_dtype_is_numeric and hdf5_dtype_is_complex and
            hdf5_dtype.itemsize != 2 * int(xml_width)/8):
        warning_channel.log(f'The metadata field {full_h5_ds_path}'
                            f' has data type "{hdf5_dtype}" but the width of'
                            ' the corresponding XML entry is set to'
                            f' "{xml_width}"')

    if verbose:
        if (xml_dtype == 'string' and xml_length != 0 and
                hdf5_dtype.kind == 'S' and
                xml_length != hdf5_dtype.itemsize):
            warning_channel.log(f'The metadata field {full_h5_ds_path}'
                                f' has data type "{hdf5_dtype}" (i.e.,'
                                'fixed-length string with'
                                f' {hdf5_dtype.itemsize}'
                                ' characters) but XML-entry length set to'
                                f' {xml_length}')


def write_xml_spec_unit_to_h5_dataset(xml_metadata_entry, h5_dataset_obj):
    """
    Write a physical unit to an HDF5 Dataset based on the
    product specification XML

    Parameters
    ----------
    xml_metadata_entry: xml.etree.ElementTree
        ElementTree object parsed from the product specifications XML file
    h5_dataset_obj: h5py.Dataset
        Product h5py dataset
    """
    warning_channel = journal.warning('write_xml_spec_unit_to_h5_dataset')
    error_channel = journal.error('write_xml_spec_unit_to_h5_dataset')

    full_h5_ds_path = xml_metadata_entry.attrib['name']

    flag_found_units = False

    # iterate over the annotation elements
    for annotation_et in xml_metadata_entry:

        # units are provided in annotation entries with attribute
        # "app" set to "conformace"
        if ('app' not in annotation_et.attrib or
                annotation_et.attrib['app'] != 'conformance'):
            continue

        # iterate over annotation attributes and locate `units`
        # (if existing)
        for annotation_key, annotation_value in \
                annotation_et.items():

            # if found multiple `units` annotation, raise an error
            if flag_found_units and annotation_key == 'units':
                error_message = ('ERROR multiple units found for'
                                 f' metadata field {full_h5_ds_path}')
                error_channel.log(error_message)
                raise ValueError(error_message)

            if annotation_key == 'units':

                # If annotation element is `units`, verify if
                # corresponding unit has a mapping to CF
                # conventions units using the dictionary
                # `cf_units_dict`. If so, write the `units`
                # into the HDF5 dataset as an attribute.
                # Units found in the dictionary `cf_units_dict`
                # have precedence over existing units
                for unit_key, unit_name in cf_units_dict.items():
                    if annotation_value != unit_key:
                        continue

                    h5_dataset_obj.attrs['units'] = \
                        np.string_(unit_name)

                    flag_found_units = True
                    break

                # Units that require a reference epoch (starting
                # with "seconds since") should be
                # filled dynamically. For those fields, verify
                # if the metadata entry has an HDF5 attribute
                # named `units` and if so and it's valid, mark
                # `flag_found_units` as `True`
                if (not flag_found_units and
                        annotation_value.startswith(
                            'seconds since') and
                        'units' in h5_dataset_obj.attrs.keys() and
                        h5_dataset_obj.attrs['units']):
                    flag_found_units = True

                # If the unit is not found within the
                # dictionary `cf_units_dict`, raise a warning
                if not flag_found_units:
                    warning_channel.log(f'The metadata field {full_h5_ds_path}'
                                        ' has an invalid unit:'
                                        f' "{annotation_value}"')


def write_xml_description_to_hdf5(xml_metadata_entry, h5_dataset_obj):
    """
    Write a description to an HDF5 Dataset based on the
    product specification XML

    Parameters
    ----------
    xml_metadata_entry: xml.etree.ElementTree
        ElementTree object parsed from the product specifications XML file
    h5_dataset_obj: h5py.Dataset
        Product h5py dataset
    """
    warning_channel = journal.warning('write_xml_description_to_hdf5')
    error_channel = journal.error('write_xml_description_to_hdf5')

    full_h5_ds_path = xml_metadata_entry.attrib['name']

    flag_found_description = False
    existing_h5_description = None

    # iterate over the annotation elements
    for annotation_et in xml_metadata_entry:

        # descriptions are provided in annotation entries with attribute
        # "app" set to "conformace"
        if ('app' not in annotation_et.attrib or
                annotation_et.attrib['app'] != 'conformance'):
            continue

        if (not existing_h5_description and
                'description' in h5_dataset_obj.attrs.keys()):
            existing_h5_description = h5_dataset_obj.attrs[
                'description'].tostring().decode()

        # update the metadata field description from XML description
        xml_description = annotation_et.text

        # if found multiple descriptions, raise an error
        if flag_found_description and xml_description:
            error_message = ('ERROR multiple description entries'
                             ' found for metadata field'
                             f' {full_h5_ds_path}')
            error_channel.log(error_message)
            raise ValueError(error_message)

        # If the product specs description is valid but the
        # product has an existing destription that does not
        # match the product specs description, raise a
        # warning
        if (xml_description and existing_h5_description and
                xml_description != existing_h5_description):
            warning_channel.log('WARNING existing metadata entry description'
                                f' for metadata entry {full_h5_ds_path}'
                                f' "{existing_h5_description}" does not match'
                                ' product specification description'
                                f' "{xml_description}"')
            flag_found_description = True

        # if the XML description is valid, update metadata field
        elif xml_description and not flag_found_description:
            flag_found_description = True
            h5_dataset_obj.attrs['description'] = \
                np.string_(xml_description)

        # if the XML description is empty, raise a warning
        if not flag_found_description:
            warning_channel.log(f'The metadata field {full_h5_ds_path} has no'
                                ' description')


class BaseWriterSingleInput():
    """
    Base writer class that can be use for all NISAR products
    """

    def __init__(self, runconfig):

        # set up processing datetime
        self.processing_datetime = datetime.now()

        # read main parameters from the runconfig
        self.runconfig = runconfig
        self.cfg = runconfig.cfg
        self.input_file = self.cfg['input_file_group']['input_file_path']

        self.output_file = \
            runconfig.cfg['product_path_group']['sas_output_file']
        partial_granule_id = \
            self.cfg['primary_executable']['partial_granule_id']

        if partial_granule_id:
            self.granule_id = partial_granule_id
        else:
            self.granule_id = '(NOT SPECIFIED)'
        self.product_type = self.cfg['primary_executable']['product_type']
        self.input_product_obj = open_product(self.input_file)

        # Example: self.input_product_type = 'RSLC' whereas
        # self.input_product_hdf5_group_type = 'SLC' (old datasets)
        self.input_product_type = self.input_product_obj.productType
        self.input_product_hdf5_group_type = \
            self.input_product_obj.ProductPath.split('/')[-1]
        self.root_path = self.input_product_obj.RootPath
        self.output_product_path = (
            f'//science/{self.input_product_obj.sarBand}SAR/'
            f'{self.product_type}')
        self.input_hdf5_obj = h5py.File(self.input_file, mode='r')
        self.output_hdf5_obj = h5py.File(self.output_file, mode='a')

    def populate_metadata(self):
        """
        Main method. It calls all other methods
        to populate the product's metadata.
        """
        self.populate_identification_common()

    def populate_identification_common(self):
        """
        Populate common parameters in the identification group
        """

        self.copy_from_input(
            'identification/absoluteOrbitNumber')

        self.copy_from_input(
            'identification/trackNumber')

        self.copy_from_input(
            'identification/frameNumber')

        self.copy_from_input(
            'identification/missionId')

        # TODO: review this
        self.set_value(
            'identification/processingCenter',
            'NASA JPL')

        self.copy_from_runconfig(
            'identification/productType',
            'primary_executable/product_type')

        self.set_value(
            'identification/granuleId',
            self.granule_id)

        self.copy_from_runconfig(
            'identification/productVersion',
            'primary_executable/product_version')

        self.set_value(
            'identification/productSpecificationVersion',
            '1.1.0')

        self.copy_from_input(
            'identification/lookDirection',
            format_function=str.title)

        self.copy_from_input(
            'identification/orbitPassDirection',
            format_function=str.title)

        self.copy_from_input(
            'identification/plannedDatatakeId')

        self.copy_from_input(
            'identification/plannedObservationId')

        self.copy_from_input(
            'identification/isUrgentObservation')

        self.copy_from_input(
             'identification/diagnosticModeFlag',
             skip_if_not_present=True)

        self.set_value(
            'identification/processingDateTime',
            self.processing_datetime.strftime(
                DATE_TIME_METADATA_FORMAT))

        self.copy_from_input('identification/radarBand',
                             default=self.input_product_obj.sarBand)

        self.copy_from_input('identification/instrumentName',
                             skip_if_not_present=True)

        processing_type_runconfig = \
            self.cfg['primary_executable']['processing_type']

        if processing_type_runconfig == 'PR':
            processing_type = np.string_('NOMINAL')
        elif processing_type_runconfig == 'UR':
            processing_type = np.string_('URGENT')
        else:
            processing_type = np.string_('UNDEFINED')
        self.set_value(
            'identification/processingType',
            processing_type,
            format_function=str.title)

        self.copy_from_input('identification/isDithered', default=False)
        self.copy_from_input('identification/isMixedMode', default=False)

    def set_value(self, h5_field, data, default=None, format_function=None):
        """
        Create an HDF5 dataset with a value set by the user

        Parameters
        ----------
        h5_field: str
            Path to the HDF5 dataset to create
        data: scalar
            Value to be assigned to the HDF5 dataset to be created
        default: scalar
            Default value to be used when input data is None
        format_function: function, optional
            Function to format string values
        """

        path_dataset_in_h5 = self.root_path + '/' + h5_field
        path_dataset_in_h5 = \
            path_dataset_in_h5.replace('{PRODUCT}', self.product_type)

        if data is None:
            data = default

        # if `data` is a numpy fixed-length string, remove trailing null
        # characters
        if ((isinstance(data, np.bytes_) or isinstance(data, np.ndarray))
                and (data.dtype.char == 'S')):
            data = np.string_(data)
            try:
                data = data.decode()
            except UnicodeDecodeError:
                pass

            # remove null characters at the right side
            data = data.rstrip('\x00')

        # if `data` is a numpy array and its data type character is
        # "O" (object), convert it to string
        elif (isinstance(data, np.ndarray) and data.dtype.char == 'O'):
            data = str(data)

        if format_function is not None:
            data = format_function(data)

        if isinstance(data, str):

            self.output_hdf5_obj.create_dataset(
                path_dataset_in_h5, data=np.string_(data))
            return

        if isinstance(data, bool):
            self.output_hdf5_obj.create_dataset(
                path_dataset_in_h5, data=np.string_(str(data)))
            return

        if (isinstance(data, list) and
                all(isinstance(item, str) for item in data)):
            self.output_hdf5_obj.create_dataset(
                path_dataset_in_h5, data=np.bytes_(data))
            return

        self.output_hdf5_obj.create_dataset(path_dataset_in_h5, data=data)

    def _copy_group_from_input(self, h5_group, *args, **kwargs):
        """
        Copy HDF5 group from the input product to the output product.

        Parameters
        ----------
        h5_group: str
            Path to the HDF5 group to copy
        """

        input_h5_field_path = self.root_path + '/' + h5_group
        output_path_dataset_in_h5 = self.root_path + '/' + h5_group

        input_h5_field_path = \
            input_h5_field_path.replace(
                '{PRODUCT}', self.input_product_hdf5_group_type)
        output_path_dataset_in_h5 = \
            output_path_dataset_in_h5.replace('{PRODUCT}',
                                              self.product_type)

        cp_h5_meta_data(self.input_hdf5_obj, self.output_hdf5_obj,
                        input_h5_field_path, output_path_dataset_in_h5,
                        *args, **kwargs)

    def copy_from_input(self, output_h5_field, input_h5_field=None,
                        default=None, skip_if_not_present=False,
                        **kwargs):
        """
        Copy HDF5 dataset value from the input product to the output
        product.

        Parameters
        ----------
        output_h5_field: str
            Path to the output HDF5 dataset to create
        input_h5_field: str, optional
            Path to the input HDF5 dataset. If not provided, the
            same path of the output dataset `output_h5_field` will
            be used
        default: scalar, optional
            Default value to be used when the input file does not
            have the dataset provided as `output_h5_field`
        skip_if_not_present: bool, optional
            Flag to prevent the execution to stop if the dataset
            is not present from input
        """
        if input_h5_field is None:
            input_h5_field = output_h5_field

        input_h5_field_path = self.root_path + '/' + input_h5_field
        input_h5_field_path = \
            input_h5_field_path.replace(
                '{PRODUCT}', self.input_product_hdf5_group_type)

        # check if the dataset is not present in the input product
        if input_h5_field_path not in self.input_hdf5_obj:

            # if the dataset is not present in the input product and
            # a default value was not provided and the flag
            # `skip_if_not_present` is set, print a warning and skip
            if (default is None and skip_if_not_present):
                warnings.warn('Metadata entry not found in the input'
                              ' product: ' + input_h5_field_path)
                return

            # if the dataset is not present in the input product and
            # a default value was not provided, raise an error
            if default is None:
                raise KeyError('Metadata entry not found in the input'
                               ' product: ' + input_h5_field_path)

            # if the dataset is not present in the input product and
            # a default value is provided, assign the default value to data
            else:
                data = default

        # othewise, if the dataset is present in the input product,
        # read it as the variable `h5_data_obj`
        else:
            h5_data_obj = self.input_hdf5_obj[input_h5_field_path]

            # check if dataset contains a string. If so, read it using method
            # `asstr()``
            if h5py.check_string_dtype(h5_data_obj.dtype):
                # use asstr() to read the dataset
                data = str(h5_data_obj.asstr()[...])

            # otherwise, read it directly without changing the datatype
            else:
                data = self.input_hdf5_obj[input_h5_field_path][...]

        self.set_value(output_h5_field, data=data, **kwargs)

    def copy_from_runconfig(self, h5_field,
                            runconfig_path,
                            default=None,
                            *args,
                            **kwargs):
        """
        Copy a runconfig value to the output as an HDF5 dataset.

        Parameters
        ----------
        h5_field: str
            Path to the output HDF5 dataset to create
        runconfig_path: str
            Path to the runconfig file
        default: scalar
            Default value to be used when the runconfig value is None
        """

        if not runconfig_path:
            raise ValueError('Please provide a valid runconfig path')

        data = self.cfg
        for key in runconfig_path.split('/'):
            data = data[key]

        if data is None:
            data = default

        self.set_value(h5_field, data=data, *args, **kwargs)

    def check_and_decorate_product_using_specs_xml(self, specs_xml_file,
                                                   verbose=False):
        """
        Check data type and decorate units and description based on a
        product specifications XML file.

        Parameters
        ----------
        specs_xml_file: str
            Product specfications XML file
        """

        specs = ET.ElementTree(file=specs_xml_file)

        # update product root attributes
        annotation_et = specs.find('./product/science/annotation')
        for key, value in annotation_et.items():
            self.output_hdf5_obj.attrs[key] = np.string_(value)

        # iterate over all XML specs parameters
        nodes_et = specs.find('./product/science/nodes')
        for xml_metadata_entry in nodes_et:
            # NISAR specs parameters are identified by the XML attribute
            # `name`
            full_h5_ds_path = xml_metadata_entry.attrib['name']
            # skip if XML attribute does not exist in the product
            if full_h5_ds_path not in self.output_hdf5_obj:
                if verbose:
                    warning_channel = journal.warning(
                        'check_and_decorate_product_using_specs_xml')
                    warning_channel.log('Dataset not found in the output'
                                        f' product: {full_h5_ds_path}')
                continue

            # otherwise, locate the XML attribute within the product
            h5_dataset_obj = self.output_hdf5_obj[full_h5_ds_path]

            check_h5_dtype_vs_xml_spec(xml_metadata_entry, h5_dataset_obj)
            write_xml_spec_unit_to_h5_dataset(xml_metadata_entry,
                                              h5_dataset_obj)
            write_xml_description_to_hdf5(xml_metadata_entry, h5_dataset_obj)

    def __enter__(self):
        return self

    def __exit__(self, ex_type, ex_value, ex_traceback):
        """
        Close open files
        """
        self.input_hdf5_obj.close()
        self.output_hdf5_obj.close()
        info_channel = journal.info('BaseWriterSingleInput')
        info_channel.log(f'File saved: {self.output_file}')
