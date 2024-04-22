#include "GeoGridProduct.h"
#include <isce3/product/RadarGridProduct.h>
#include <isce3/product/Serialization.h>

/** @param[in] file IH5File object for product. */
isce3::product::GeoGridProduct::
GeoGridProduct(isce3::io::IH5File & file) {

    // read base group "/science"
    std::string base_dir = "/science/";
    isce3::io::IGroup base_group = file.openGroup(base_dir);

    // Locate the identificaton group
    auto identification_vector = isce3::product::findGroupPath(base_group, "identification");
    if (identification_vector.size() == 0) {
        std::string error_msg = ("ERROR identification group not found in " +
                                 file.getFileName());
        throw isce3::except::RuntimeError(ISCE_SRCINFO(), error_msg);
    } else if (identification_vector.size() > 1) {
        std::string error_msg = ("ERROR there should be only one identification"
                                 " group in " +
                                 file.getFileName());
        throw isce3::except::RuntimeError(ISCE_SRCINFO(), error_msg);
    }

    // set look side (direction)
    std::string identification_group_str = base_dir + identification_vector[0];
    std::string look_dir;
    isce3::io::loadFromH5(
            file, identification_group_str + "/lookDirection", look_dir);
    lookSide(look_dir);

     // set product type
    std::string product_type_str;
    isce3::io::loadFromH5(
            file, identification_group_str + "/productType", product_type_str);

    std::vector<std::string> key_vector = {"grids"};

    std::string image_group_str = "", metadata_group_str;
    setImageMetadataGroupStr(file, base_dir, base_group, key_vector,
        image_group_str, metadata_group_str);

    // If did not find HDF5 groups grids
    if (image_group_str.size() == 0) {
        std::string error_msg = ("ERROR grids groups not found in " +
                                 file.getFileName());
        throw isce3::except::RuntimeError(ISCE_SRCINFO(), error_msg);
    }

    // Get grids group
    isce3::io::IGroup imGroup = file.openGroup(image_group_str);

    // Configure grids
    loadFromH5(imGroup, _grids);

    // Get metadata group
    isce3::io::IGroup metaGroup = file.openGroup(metadata_group_str);

    // Some GCOV grids metadata is stored under "metadata/sourceData"
    if (product_type_str == "GCOV" and isce3::io::exists(metaGroup, "sourceData/swaths")) {

        // Get sourceData group
        std::string swaths_group_str = metadata_group_str + "/sourceData/swaths";
        isce3::io::IGroup swaths_group = file.openGroup(swaths_group_str);

        // Update grids object with GCOV sourceData group
        if (isce3::io::exists(swaths_group, "frequencyA")) {
            populateGridSwathParameterFromH5(swaths_group, _grids['A'], 'A');
        }
        if (isce3::io::exists(swaths_group, "frequencyB")) {
            populateGridSwathParameterFromH5(swaths_group, _grids['B'], 'B');
        }
    }

    // Configure metadata
    loadFromH5(metaGroup, _metadata, "L2");

    // Save the filename
    _filename = file.filename();
}
