Author's Note:
(Joshua Cohen)

This subset of the isce::core library is built to be optimized (or at the very least compile-able
by nvcc) for GPU execution and for interfacing with the rest of the isce::core library. Given
the in-flux nature of the development of the GPU algorithms, this library is far from final, and
is only meant to be an additional set of buildable object for inclusion into the CUDA C++ codebase.

The primary question for the future (relative to this portion of the library at least) is whether
we can somehow redesign the isce::core library to build both the CPU-optimized library together
with the GPU-optimized library without needing to maintain separate codes. The main issue with
doing this at the moment is the dissimilarities between the supported C++ implementations in the
CUDA APIs vs the native C++ implementations. There are tricks to making this code compile without
complaining using just g++ but it's a pain and tends to make the code fairly unreadable. We need
to do some more serious thinking about the best way to accomplish this...
