import h5py

class H5pyGroupWrapper:
    '''
    h5py group wrapper class that allows overwritting HDF5 datasets if
    they already exist. This class is helpful for populating
    HDF5 templates in which the HDF5 datasets already exist and
    we don't want to remove all datasets before adding new values.
    '''

    def __init__(self, h5_group):
        self.h5_group = h5_group

    def create_dataset(self, key, **kwargs):
        '''
        Wrapper for creating dataset. If dataset exists
        delete it before calling create_dataset()
        '''
        if key in self.h5_group:
            del self.h5_group[key]
        ret = self.h5_group.create_dataset(key, **kwargs)
        return ret

    def create_group(self, group_name, overwrite = False):
        '''
        Wrapper for creating group. By default
        does not overwrite group. Unless flag
        overwrite is set. 
        '''
        if group_name in self.h5_group and overwrite:
            del self.h5_group[group_name]
        if group_name in self.h5_group:
            ret = self.h5_group[group_name]
        else:
            ret = self.h5_group.create_group(group_name)
        new_group = H5pyGroupWrapper(ret)
        return new_group

    def __getattr__(self, name):
        '''
        Forward call to original object
        '''
        return getattr(self.h5_group, name)

    def __iter__(self):
        '''
        Forward call to original object
        '''
        return self.h5_group.__iter__()

    def __getitem__(self, key):
        '''
        Wrapper for getitem. 
        If item is a dataset, return the original object.
        Otherwise, return wrapped group.
        '''
        ret = self.h5_group.__getitem__(key)
        if isinstance(ret, h5py.Dataset):
            return ret
        new_group = H5pyGroupWrapper(ret)
        return new_group

    def __setitem__(self, key, value):
        '''
        Wrapper for setitem. 
        If item is not present, add it according to the value type.
        Othewise, do the same, but keeping the original item type 
        (dataset or group)
        '''
        if (key not in self.h5_group and 
               isinstance(value, h5py.Group)):
            ret = self.create_group(key, value)
            new_group = H5pyGroupWrapper(ret)
            return new_group 
        if key not in self.h5_group:
            ret = self.create_dataset(key, data = value)
            return ret 
        ret = self.h5_group[key]
        if isinstance(ret, h5py.Dataset):
            ret = self.create_dataset(key, data = value)
            return ret
        ret = self.create_group(key, value)
        new_group = H5pyGroupWrapper(ret)
        return new_group

    def __delitem__(self, key):
        '''
        Wrapper for deleting an item
        Only deletes if the item exists
        '''
        if key not in self.h5_group:
            return
        del self.h5_group[key]