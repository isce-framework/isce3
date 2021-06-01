#include "Product.h"
#include <isce3/product/Serialization.h>

/** Find unique group path excluding repeated occurrences
*/
std::vector<std::string> _findGroupPath(isce3::io::IGroup& group, 
                                        const std::string group_name) {
    
    auto group_vector = group.find(group_name, ".", "GROUP");

    std::vector<std::string> filtered_group_vector;

    // Filter unique group paths
    for (const auto& full_group_path: group_vector) {
        std::size_t group_name_position = full_group_path.find(group_name);

        // Get base group path (remove sub-groups)
        std::string base_group = full_group_path.substr(
            0, group_name_position + group_name.length());

        /* If base group is not in filtered_group_vector, 
        append it to the vector
        */
        if (find(filtered_group_vector.begin(), 
                 filtered_group_vector.end(), 
                 base_group) ==  filtered_group_vector.end()) {
            filtered_group_vector.push_back(base_group);
            }
    }

    return filtered_group_vector;
}


/** @param[in] file IH5File object for product. */
isce3::product::Product::
Product(isce3::io::IH5File & file) {

    std::string base_dir = "/science/";

    isce3::io::IGroup base_group = file.openGroup(base_dir);

    bool flag_has_swaths = false;

    /* 
    In this section, we look for grids or swaths group within base_group.
    We start by assigning an empty string to image_group_str in case
    grids and swaths group are not found. 
    */
   std::vector<std::string> key_vector = {"grids", "swaths"};
    std::string image_group_str = "", metadata_group_str; 
    for (const auto& key: key_vector) {

        // Look for HDF5 groups that match key (i.e., "grids" or "swaths")
        auto group_vector = _findGroupPath(base_group, key);

        if (group_vector.size() > 1) {
            /*
            NISAR products should have only one grids or swaths group
            within base_group:
            */ 
            std::string error_msg = ("ERROR there should be at most one " +
                                     key + " group in " +file.getFileName());
            throw isce3::except::RuntimeError(ISCE_SRCINFO(), error_msg);
        }
        else if (group_vector.size() > 0) {

            // Group swaths or grid was found

            /* 
            Assign the group (grids or swaths) to image_group_str,
            locate the group metadata by substituting "grids"/"swaths" with
            "metadata", and save it on metadata_group_str
            */
            image_group_str = base_dir + group_vector[0];
            metadata_group_str = image_group_str;
            std::size_t key_position = metadata_group_str.rfind(key);
            metadata_group_str.replace(key_position, key.length(), "metadata");
            if (key == "swaths") {
                /* 
                flag_has_swaths indicates if image_group_str reffers to swaths or
                grids
                */
                flag_has_swaths = true;
            }
            break;
        } 
    }

    // If did not find HDF5 groups swaths or grids
    if (image_group_str.size() == 0) {
        std::string error_msg = ("ERROR swaths and grids groups"
                                 " not found in " + file.getFileName());
        throw isce3::except::RuntimeError(ISCE_SRCINFO(), error_msg);
    }

    // Get swaths/grids group
    isce3::io::IGroup imGroup = file.openGroup(image_group_str);

    // Configure swaths
    if (flag_has_swaths) {
        loadFromH5(imGroup, _swaths);
    } 
    /* 
    Not implemented yet:
    else {
        loadFromH5(imGroup, _grids); 
    }
    */

    // Get metadata group
    isce3::io::IGroup metaGroup = file.openGroup(metadata_group_str); 
    // Configure metadata

    loadFromH5(metaGroup, _metadata);
    
    // Get look direction
    auto identification_vector = _findGroupPath(
        base_group, "identification");
    if (identification_vector.size() == 0) {
            std::string error_msg = ("ERROR identification group not found in " +
                                     file.getFileName());
            throw isce3::except::RuntimeError(ISCE_SRCINFO(), error_msg);
    }
    else if (identification_vector.size() > 1) {
            std::string error_msg = ("ERROR there should be only one identification"
                                     " group in " + file.getFileName());
            throw isce3::except::RuntimeError(ISCE_SRCINFO(), error_msg);
    }
    
    std::string identification_group_str = base_dir + identification_vector[0];
    std::string lookDir;
    isce3::io::loadFromH5(file, identification_group_str + "/lookDirection", 
                          lookDir);
    lookSide(lookDir);

    // Save the filename
    _filename = file.filename();

}
