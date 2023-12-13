import h5py
import warnings
import numpy as np
from datetime import datetime

from nisar.products.readers import open_product

from nisar.h5 import cp_h5_meta_data

DATE_TIME_METADATA_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'


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
        # self.input_slc_product_path = slc.ProductPath
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
            '0.9.0')

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

        # TODO: fix this
        self.set_value(
             'identification/diagnosticModeFlag',
             '(NOT SPECIFIED)')

        self.set_value(
            'identification/processingDateTime',
            self.processing_datetime.strftime(
                DATE_TIME_METADATA_FORMAT))

        self.copy_from_input('identification/radarBand',
                             default=self.input_product_obj.sarBand)

        self.copy_from_input('identification/instrumentName',
                             default='(NOT SPECIFIED)')

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

        if isinstance(data, list) and \
            all(isinstance(item, str) for item in data):
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

        try:
            data = self.input_hdf5_obj[input_h5_field_path][...]
        except KeyError:
            # if a default value was not provided and flag
            # `skip_if_not_present`, skip
            if default is None and skip_if_not_present:
                warnings.warn('Invalid key for the input product: ' +
                              input_h5_field_path)
                return

            # otherwise, if a default value was not provided, raise an error
            elif default is None:
                raise KeyError('Invalid key for the input product: ' +
                               input_h5_field_path)

            # otherwise, assign the default value to data
            data = default

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

    def __enter__(self):
        return self

    def __exit__(self, ex_type, ex_value, ex_traceback):
        """
        Close open files
        """
        self.input_hdf5_obj.close()
        self.output_hdf5_obj.close()
