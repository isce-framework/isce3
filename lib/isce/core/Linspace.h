#pragma once
#ifndef ISCE_CORE_LINSPACE_H
#define ISCE_CORE_LINSPACE_H

#include <stddef.h>

#include "Common.h"

namespace isce { namespace core {

/** A uniformly-spaced sequence of values over some interval. */
template<typename T>
class Linspace {
public:
    /**
     * Construct a Linspace over the closed interval [first, last].
     *
     * The behavior is undefined if size <= 1.
     *
     * @param[in] first first sample in interval
     * @param[in] last last sample in interval
     * @param[in] size number of samples
     */
    CUDA_HOSTDEV
    static
    Linspace<T> from_interval(T first, T last, int size);

    Linspace() = default;

    /**
     * Constructor
     *
     * @param[in] first first sample
     * @param[in] spacing sample spacing
     * @param[in] size number of samples
     */
    CUDA_HOSTDEV
    constexpr
    Linspace(T first, T spacing, int size);

    /** Copy constructor */
    template<typename U>
    CUDA_HOSTDEV
    Linspace(const Linspace<U> &);

    /** Assign values. */
    template<typename U>
    CUDA_HOSTDEV
    Linspace<T> & operator=(const Linspace<U> &);

    /**
     * Return sample at the specified position.
     *
     * The behavior is undefined if pos is out of range (i.e. if it is >= size()).
     */
    CUDA_HOSTDEV
    constexpr
    T operator[](int pos) const;

    /** First sample */
    CUDA_HOSTDEV
    constexpr
    T first() const;

    /** Last sample */
    CUDA_HOSTDEV
    constexpr
    T last() const;

    /** Sample spacing */
    CUDA_HOSTDEV
    constexpr
    T spacing() const;

    /** Number of samples */
    CUDA_HOSTDEV
    constexpr
    int size() const;

    /**
     * Return a sub-Linspace over the half-open interval [start, stop).
     *
     * The behavior is undefined if [start, stop) is not a valid interval
     * (i.e. start > stop) or if start or stop are out of range
     * (i.e. if start >= size() or stop > size()).
     *
     * @param[in] start start position
     * @param[in] stop end position (not included in interval)
     */
    CUDA_HOSTDEV
    Linspace<T> subinterval(int start, int stop) const;

    /** Check if the sequence contains no samples. */
    CUDA_HOSTDEV
    constexpr
    bool empty() const;

private:
    T _first;
    T _spacing;
    int _size;
};

template<typename T, typename U>
CUDA_HOSTDEV
bool operator==(const Linspace<T> &, const Linspace<U> &);

template<typename T, typename U>
CUDA_HOSTDEV
bool operator!=(const Linspace<T> &, const Linspace<U> &);

/**
 * Return the position where the specified value would be inserted in the
 * sequence in order to maintain sorted order.
 */
template<typename T, typename U>
CUDA_HOSTDEV
int where(const Linspace<T> & x, U val);

}}

#define ISCE_CORE_LINSPACE_ICC
#include "Linspace.icc"
#undef ISCE_CORE_LINSPACE_ICC

#endif

