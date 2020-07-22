#include "Product.h"
#include <isce3/product/Serialization.h>

/** @param[in] file IH5File object for product. */
isce3::product::Product::
Product(isce3::io::IH5File & file) {
    // Get swaths group
    isce3::io::IGroup imGroup = file.openGroup("/science/LSAR/SLC/swaths");
    // Configure swaths
    loadFromH5(imGroup, _swaths);
    // Get metadata group
    isce3::io::IGroup metaGroup = file.openGroup("/science/LSAR/SLC/metadata"); 
    // Configure metadata
    loadFromH5(metaGroup, _metadata);
    // Get look direction
    std::string lookDir;
    isce3::io::loadFromH5(file, "/science/LSAR/identification/lookDirection", lookDir);
    lookSide(lookDir);
    // Save the filename
    _filename = file.filename();
}
