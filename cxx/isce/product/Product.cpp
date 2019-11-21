#include "Product.h"
#include <isce/product/Serialization.h>

/** @param[in] file IH5File object for product. */
isce::product::Product::
Product(isce::io::IH5File & file) {
    // Get swaths group
    isce::io::IGroup imGroup = file.openGroup("/science/LSAR/SLC/swaths");
    // Configure swaths
    loadFromH5(imGroup, _swaths);
    // Get metadata group
    isce::io::IGroup metaGroup = file.openGroup("/science/LSAR/SLC/metadata"); 
    // Configure metadata
    loadFromH5(metaGroup, _metadata);
    // Get look direction
    std::string lookDir;
    isce::io::loadFromH5(file, "/science/LSAR/identification/lookDirection", lookDir);
    lookSide(lookDir);
    // Save the filename
    _filename = file.filename();
}
