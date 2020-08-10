#pragma once

#include <iostream>
#include <string>

namespace isce3 { namespace cuda { namespace core {

/**
 * CUDA device compute capability
 *
 * ComputeCapability identifies a CUDA device's architecture generation and
 * feature compatibility.
 */
struct ComputeCapability {
    /**
     * Construct a new ComputeCapability object.
     *
     * \param[in] major Major compute version
     * \param[in] minor Minor compute version
     */
    constexpr ComputeCapability(int major, int minor)
        : major {major}, minor {minor}
    {}

    explicit operator std::string() const
    {
        return std::to_string(major) + "." + std::to_string(minor);
    }

    friend std::ostream& operator<<(std::ostream& os, ComputeCapability cc)
    {
        return os << std::string(cc);
    }

    friend constexpr bool operator==(ComputeCapability lhs,
                                     ComputeCapability rhs) noexcept
    {
        return lhs.major == rhs.major and lhs.minor == rhs.minor;
    }

    friend constexpr bool operator!=(ComputeCapability lhs,
                                     ComputeCapability rhs) noexcept
    {
        return not(lhs == rhs);
    }

    friend constexpr bool operator<(ComputeCapability lhs,
                                    ComputeCapability rhs) noexcept
    {
        return lhs.major < rhs.major or
               (lhs.major == rhs.major and lhs.minor < rhs.minor);
    }

    friend constexpr bool operator>(ComputeCapability lhs,
                                    ComputeCapability rhs) noexcept
    {
        return lhs.major > rhs.major or
               (lhs.major == rhs.major and lhs.minor > rhs.minor);
    }

    friend constexpr bool operator<=(ComputeCapability lhs,
                                     ComputeCapability rhs) noexcept
    {
        return not(lhs > rhs);
    }

    friend constexpr bool operator>=(ComputeCapability lhs,
                                     ComputeCapability rhs) noexcept
    {
        return not(lhs < rhs);
    }

    int major; /**< Major compute version */
    int minor; /**< Minor compute version */
};

/** Return the library's minimum supported compute capability. */
constexpr ComputeCapability minComputeCapability() noexcept { return {3, 5}; }

}}} // namespace isce3::cuda::core
