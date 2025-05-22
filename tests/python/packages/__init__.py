# This __init__ file is not used to import anything; its purpose is to create a package
# called `packages` that contains the `isce3` subpackage. This prevents a bug where the
# `isce3` package containing the tests would shadow the true `isce3` package while
# running the test suite, preventing the latter from being imported by the tests.