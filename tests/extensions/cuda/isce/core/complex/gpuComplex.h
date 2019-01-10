#include <valarray>
#include <complex>
#include <random>

using std::valarray;
using std::complex;

template <class T>
void makeRandomReal(valarray<T> &random_data, size_t n_pts) {
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-100.,100.);
    for (int i=0; i<n_pts; ++i) {
        random_data[i] = distribution(generator);
    }
}

template <class T>
void makeRandomStdComplex(valarray<complex<T>> &random_data, size_t n_pts) {
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-100.,100.);
    for (int i=0; i<n_pts; ++i) {
        T real = distribution(generator);
        T imag = distribution(generator);
        random_data[i] = complex<T>(real, imag);
    }
}

