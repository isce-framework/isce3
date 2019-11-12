#pragma once

#include "Common.h"

namespace isce { namespace core {

/** A uniformly-spaced sequence of values over some interval. */
template<typename T>
class Linspace {
public:

    /**
     * Construct a Linspace over the closed interval [\p first, \p last].
     *
     * The behavior is undefined if \p size <= 1.
     *
     * \param[in] first first sample in interval
     * \param[in] last last sample in interval
     * \param[in] size number of samples
     */
    CUDA_HOSTDEV
    constexpr
    static
    Linspace<T> from_interval(T first, T last, int size);

    Linspace() = default;

    /**
     * Constructor
     *
     * \param[in] first first sample
     * \param[in] spacing sample spacing
     * \param[in] size number of samples
     */
    CUDA_HOSTDEV
    constexpr
    Linspace(T first, T spacing, int size);

    template<typename U>
    CUDA_HOSTDEV
    constexpr
    Linspace(const Linspace<U> &);

    template<typename U>
    CUDA_HOSTDEV
    constexpr
    Linspace<T> & operator=(const Linspace<U> &);

    /**
     * Return sample at the specified position.
     *
     * The behavior is undefined if \p pos is out of range.
     */
    CUDA_HOSTDEV
    constexpr
    T operator[](int pos) const { return _first + pos * _spacing; }

    /** First sample */
    CUDA_HOSTDEV
    constexpr
    T first() const { return _first; }

    /** Set first sample */
    CUDA_HOSTDEV
    constexpr
    void first(T first) { _first = first; }

    /** Last sample */
    CUDA_HOSTDEV
    constexpr
    T last() const { return operator[](_size - 1); }

    /** Sample spacing */
    CUDA_HOSTDEV
    constexpr
    T spacing() const { return _spacing; }

    /** Set Sample Spacing */
    CUDA_HOSTDEV
    constexpr
    void spacing(T spacing) { _spacing = spacing; };

    /** Number of samples */
    CUDA_HOSTDEV
    constexpr
    int size() const { return _size; }

    /**
     * Change the number of samples in the sequence.
     *
     * \param[in] size new size
     */
    CUDA_HOSTDEV
    constexpr
    void resize(int size);

    /**
     * Return a sub-Linspace over the half-open interval [\p start, \p stop).
     *
     * The behavior is undefined if \p start or \p stop are out-of-range
     *
     * \param[in] start start position
     * \param[in] stop end position
     */
    CUDA_HOSTDEV
    constexpr
    Linspace<T> subinterval(int start, int stop) const;

    /** Check if the sequence contains no samples. */
    CUDA_HOSTDEV
    constexpr
    bool empty() const { return _size == 0; }

    /**
     * Return the position where the specified value would be inserted in the
     * sequence in order to maintain sorted order.
     */
    template<typename U>
    CUDA_HOSTDEV
    constexpr
    int search(U) const;

private:
    T _first = {};
    T _spacing = {};
    int _size = 0;
};

template<typename T, typename U>
CUDA_HOSTDEV
constexpr
bool operator==(const Linspace<T> &, const Linspace<U> &);

template<typename T, typename U>
CUDA_HOSTDEV
constexpr
bool operator!=(const Linspace<T> &, const Linspace<U> &);

}}

#define ISCE_CORE_LINSPACE_ICC
#include "Linspace.icc"
#undef ISCE_CORE_LINSPACE_ICC
