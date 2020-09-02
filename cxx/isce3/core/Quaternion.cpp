#include "Quaternion.h"

#include "EulerAngles.h"

namespace isce3 { namespace core {

// Not in header to break circular dependence between Quaternion and
// EulerAngles.
Quaternion::Quaternion(const EulerAngles& ypr) : super_t(ypr.toRotationMatrix())
{}

}} // namespace isce3::core
