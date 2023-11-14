#pragma once

#include <vector>

namespace isce3::core::detail {

/** Small-size optimized (SSO) buffer
 *
 * @tparam T data type of buffer
 * @tparam NMAX Largest number of elements to allocate on the stack.
 */
template<typename T, size_t NMAX = 32>
class SSOBuffer {
public:
    /** Constructor
     *
     * @param[in]   n   Size of buffer.  If n > NMAX then memory will be
     *                  allocated on the heap.
     */
    SSOBuffer(size_t n) : size_ {n}
    {
        if (n <= NMAX) {
            ptr_ = stackbuf_;
        } else {
            heapbuf_.resize(n);
            ptr_ = heapbuf_.data();
        }
    }

    /** Number of elements that can be stored in the buffer */
    size_t size() const { return size_; }

    /** const pointer to the underlying array serving as element storage */
    const T* data() const { return ptr_; }

    /** pointer to the underlying array serving as element storage */
    T* data() { return ptr_; }

    /** Index (read).  No bounds check. */
    const T& operator[](size_t i) const { return ptr_[i]; }

    /** Index (write).  No bounds check. */
    T& operator[](size_t i) { return ptr_[i]; }

    // Don't bother implementing copy & assignment, so delete default ones
    SSOBuffer(const SSOBuffer&) = delete;
    SSOBuffer& operator= (const SSOBuffer&) = delete;

private:
    T stackbuf_[NMAX];
    std::vector<T> heapbuf_;
    T* ptr_;
    const size_t size_;
};

} // namespace isce3::core::detail
