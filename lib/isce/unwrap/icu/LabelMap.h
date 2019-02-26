#ifndef ISCE_UNWRAP_ICU_LABELMAP_H
#define ISCE_UNWRAP_ICU_LABELMAP_H

#include <cstddef> // size_t
#include <cstdint> // uint8_t
#include <vector> // std::vector

namespace isce { namespace unwrap { namespace icu {

// \brief Table of connected component label equivalences
//
// Maintains a list of all connected component labels along with a mapping 
// to their minimum equivalent label.
class LabelMap
{
public:
    // Constructor
    LabelMap();
    // Add new label to table and return the label.
    uint8_t nextlabel();
    // Get mapped label.
    uint8_t getlabel(const uint8_t) const;
    // Update a label mapping.
    void setlabel(const uint8_t oldlabel, const uint8_t newlabel);
    // Get number of labels.
    size_t size() const;

private:
    // Table of label mappings.
    std::vector<uint8_t> _labels;
};

} } }

// Get inline implementations.
#define ISCE_UNWRAP_ICU_LABELMAP_ICC
#include "LabelMap.icc"
#undef ISCE_UNWRAP_ICU_LABELMAP_ICC

#endif /* ISCE_UNWRAP_ICU_LABELMAP_H */

