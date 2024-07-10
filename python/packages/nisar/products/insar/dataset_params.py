from dataclasses import dataclass, field

import numpy as np


@dataclass
class DatasetParams:
    """
    Convenience dataclass for passing dataset parameters to be written to
    h5py.Dataset

    Attributes
    ----------
    name : str
        Dataset name
    value : object
        Data to be stored in Dataset
    description : str
        Description attribute of Dataset. Could be in attr_dict but made
        independent member to highlight it as a requirement.
    attr_dict : dict
        Other attributes to be written to Dataset
    """
    name: str
    value: object
    description: str
    attr_dict: dict = field(default_factory=dict)

def add_dataset_and_attrs(group, dataset_param_item):
    """
    Write a DatasetParam object to h5py.Group

    Parameters
    ----------
    group : h5py.Group
        h5py Group to store the  dataset_param_item
    dataset_param_item : DatasetParams
        DatasetParams object to write to group
    """
    # Ensure it is clear to write by deleting pre-existing Dataset
    if dataset_param_item.name in group:
        del group[dataset_param_item.name]

    def _as_np_string_if_needed(val):
        """
        Internal convenience function where if type str encountered, convert
        and return as np.bytes_. Otherwise return as is.
        """
        val = np.bytes_(val) if isinstance(val, str) else val
        return val

    # Convert data to written if necessary
    val = _as_np_string_if_needed(dataset_param_item.value)
    try:
        if val is None:
            group[dataset_param_item.name] = np.nan
        else:
            group[dataset_param_item.name] = val
    except TypeError:
        raise TypeError(f"unable to write {dataset_param_item.name}")

    # Write data to dataset
    val_ds = group[dataset_param_item.name]
    desc = _as_np_string_if_needed(dataset_param_item.description)
    val_ds.attrs["description"] = desc

    # Write attributes (if any)
    for key, val in dataset_param_item.attr_dict.items():
        val_ds.attrs[key] = _as_np_string_if_needed(val)
