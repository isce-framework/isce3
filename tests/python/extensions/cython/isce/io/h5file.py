#!/usr/bin/env python3
from isce3.extensions.isceextension import pyIH5File, H5FileIException

def testFileNotFound():
    try:
        pyIH5File("some_file_name_that_doesnt_exist")
    except H5FileIException:
        pass # good, that's what we expected to happen

    # let any other exceptions fall through

    else:
        raise RuntimeError("opening a weird filename... succeeded!?")

if __name__ == '__main__':
    testFileNotFound()
