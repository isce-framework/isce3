#include "EulerAngles.h"

#include "Quaternion.h"

namespace isce3 { namespace core {

// Not in header to break circular dependence between EulerAngles and
// Quaternion.
EulerAngles::EulerAngles(const Quaternion& q)
    : EulerAngles(q.toRotationMatrix())
{}

}} // namespace isce3::core
