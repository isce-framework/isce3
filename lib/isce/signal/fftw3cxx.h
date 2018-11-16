//  fftw3cxx.hh version 1.2
/*
 *  Copyright (c) 2017 Gregory E. Allen
 *  
 *  This is free and unencumbered software released into the public domain.
 *  
 *  Anyone is free to copy, modify, publish, use, compile, sell, or
 *  distribute this software, either in source code form or as a compiled
 *  binary, for any purpose, commercial or non-commercial, and by any
 *  means.
 *  
 *  In jurisdictions that recognize copyright laws, the author or authors
 *  of this software dedicate any and all copyright interest in the
 *  software to the public domain. We make this dedication for the benefit
 *  of the public at large and to the detriment of our heirs and
 *  successors. We intend this dedication to be an overt act of
 *  relinquishment in perpetuity of all present and future rights to this
 *  software under copyright law.
 *  
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 *  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 *  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 *  IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 *  OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 *  ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 *  OTHER DEALINGS IN THE SOFTWARE.
 */
/*
 * This file borrows significantly from fftw3.h, with the copyright below.
 */
/*
 * Copyright (c) 2003, 2007-14 Matteo Frigo
 * Copyright (c) 2003, 2007-14 Massachusetts Institute of Technology
 *
 * The following statement of license applies *only* to this header file,
 * and *not* to the other files distributed with FFTW or derived therefrom:
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef fftw3cxx_hh
#define fftw3cxx_hh
#pragma once

#include <stdexcept>
#include <complex>
#include <fftw3.h>

//---
// Between fftw releases 3.2 and 3.3.5, several new API functions were added.
// We forward declare them here, so fftw3cxx works with any of these versions.
// If these functions were also declared in fftw3.h, no harm.
//---

#define FFTW3CXX_DEFINE_FFTW330_NEW_API(X, R, C) \
FFTW_EXTERN int X(export_wisdom_to_filename)(const char *filename); \
FFTW_EXTERN void X(export_wisdom_to_file)(FILE *output_file); \
FFTW_EXTERN int X(import_wisdom_from_filename)(const char *filename); \
FFTW_EXTERN int X(import_wisdom_from_file)(FILE *input_file); \
FFTW_EXTERN R *X(alloc_real)(size_t n); \
FFTW_EXTERN C *X(alloc_complex)(size_t n); \
FFTW_EXTERN double X(cost)(const X(plan) p);

#define FFTW3CXX_DEFINE_FFTW335_NEW_API(X, R, C) \
FFTW_EXTERN char *X(sprint_plan)(const X(plan) p); \
FFTW_EXTERN int X(alignment_of)(R *p); \
FFTW_EXTERN void X(make_planner_thread_safe)(void);

extern "C" {
    FFTW3CXX_DEFINE_FFTW330_NEW_API(FFTW_MANGLE_DOUBLE, double, fftw_complex)
    FFTW3CXX_DEFINE_FFTW330_NEW_API(FFTW_MANGLE_FLOAT, float, fftwf_complex)
    FFTW3CXX_DEFINE_FFTW330_NEW_API(FFTW_MANGLE_LONG_DOUBLE, long double, fftwl_complex)
    FFTW3CXX_DEFINE_FFTW335_NEW_API(FFTW_MANGLE_DOUBLE, double, fftw_complex)
    FFTW3CXX_DEFINE_FFTW335_NEW_API(FFTW_MANGLE_FLOAT, float, fftwf_complex)
    FFTW3CXX_DEFINE_FFTW335_NEW_API(FFTW_MANGLE_LONG_DOUBLE, long double, fftwl_complex)
};

namespace isce{
namespace fftw3cxx {

//---
// Alias the fftw public API with a C++ templated-type style.
// e.g. fftw_plan becomes fftw<double>::plan
//---

/*
  Use a large macro in the style of fftw3.h
 
  X: name-mangling macro
  R: real data type
  C: complex data type
*/

#define FFTW3CXX_DEFINE_API(X, R, C) \
\
template<> \
struct fftw<R> { \
    typedef X(plan) plan; \
    typedef R real; \
    typedef X(complex) complex; \
    typedef X(iodim) iodim; \
    typedef X(iodim64) iodim64; \
    typedef X(r2r_kind) r2r_kind; \
\
    static void execute(const plan p) \
        { X(execute)(p); } \
\
    static plan plan_dft(int rank, const int *n, \
        C *in, C *out, int sign, unsigned flags) \
        { return X(plan_dft)(rank, n, in, out, sign, flags); } \
\
    static plan plan_dft_1d(int n, C *in, C *out, \
        int sign, unsigned flags) \
        { return X(plan_dft_1d)(n, in, out, sign, flags); } \
    static plan plan_dft_2d(int n0, int n1, \
        C *in, C *out, int sign, unsigned flags) \
        { return X(plan_dft_2d)(n0, n1, in, out, sign, flags); } \
    static plan plan_dft_3d(int n0, int n1, int n2, \
        C *in, C *out, int sign, unsigned flags) \
        { return X(plan_dft_3d)(n0, n1, n2, in, out, sign, flags); } \
\
    static plan plan_many_dft(int rank, const int *n, int howmany, \
        C *in, const int *inembed, int istride, int idist, \
        C *out, const int *onembed, int ostride, int odist, \
        int sign, unsigned flags) \
        { return X(plan_many_dft)(rank, n, howmany, \
            in, inembed, istride, idist, \
            out, onembed, ostride, odist, sign, flags); } \
\
    static plan plan_guru_dft(int rank, const iodim *dims, \
        int howmany_rank, const iodim *howmany_dims, \
        C *in, C *out, int sign, unsigned flags) \
        { return X(plan_guru_dft)(rank, dims, \
            howmany_rank, howmany_dims, in, out, sign, flags); } \
    static plan plan_guru_split_dft(int rank, const iodim *dims, \
        int howmany_rank, const iodim *howmany_dims, \
        R *ri, R *ii, R *ro, R *io, unsigned flags) \
        { return X(plan_guru_split_dft)(rank, dims, \
            howmany_rank, howmany_dims, ri, ii, ro, io, flags); } \
\
    static plan plan_guru64_dft(int rank, const iodim64 *dims, \
        int howmany_rank, const iodim64 *howmany_dims,\
        C *in, C *out, int sign, unsigned flags) \
        { return X(plan_guru64_dft)(rank, dims, \
            howmany_rank, howmany_dims, in, out, sign, flags); } \
    static plan plan_guru64_split_dft(int rank, const iodim64 *dims, \
        int howmany_rank, const iodim64 *howmany_dims, \
        R *ri, R *ii, R *ro, R *io, unsigned flags) \
        { return X(plan_guru64_split_dft)(rank, dims, \
            howmany_rank, howmany_dims, ri, ii, ro, io, flags); } \
\
    static void execute_dft(const plan p, C *in, C *out) \
        { X(execute_dft)(p, in, out); } \
    static void execute_split_dft(const plan p, R *ri, R *ii, \
        R *ro, R *io) \
        { X(execute_split_dft)(p, ri, ii, ro, io); } \
\
    static plan plan_many_dft_r2c(int rank, const int *n, int howmany, \
        R *in, const int *inembed, int istride, int idist, \
        C *out, const int *onembed, int ostride, int odist, \
        unsigned flags) \
        { return X(plan_many_dft_r2c)(rank, n, howmany, \
            in, inembed, istride, idist, \
            out, onembed, ostride, odist, flags); } \
\
    static plan plan_dft_r2c(int rank, const int *n, \
        R *in, C *out, unsigned flags) \
        { return X(plan_dft_r2c)(rank, n, in, out, flags); } \
\
    static plan plan_dft_r2c_1d(int n, R *in, C *out, \
        unsigned flags) \
        { return X(plan_dft_r2c_1d)(n, in, out, flags); } \
    static plan plan_dft_r2c_2d(int n0, int n1, \
        R *in, C *out, unsigned flags) \
        { return X(plan_dft_r2c_2d)(n0, n1, in, out, flags); } \
    static plan plan_dft_r2c_3d(int n0, int n1, int n2, \
        R *in, C *out, unsigned flags) \
        { return X(plan_dft_r2c_3d)(n0, n1, n2, in, out, flags); } \
\
    static plan plan_many_dft_c2r(int rank, const int *n, int howmany, \
        C *in, const int *inembed, int istride, int idist, \
        R *out, const int *onembed, int ostride, int odist, \
        unsigned flags) \
        { return X(plan_many_dft_c2r)(rank, n, howmany, \
            in, inembed, istride, idist, \
            out, onembed, ostride, odist, flags); } \
\
    static plan plan_dft_c2r(int rank, const int *n, \
        C *in, R *out, unsigned flags) \
        { return X(plan_dft_c2r)(rank, n, in, out, flags); } \
\
    static plan plan_dft_c2r_1d(int n, C *in, R *out, unsigned flags) \
        { return X(plan_dft_c2r_1d)(n, in, out, flags); } \
    static plan plan_dft_c2r_2d(int n0, int n1, \
        C *in, R *out, unsigned flags) \
        { return X(plan_dft_c2r_2d)(n0, n1, in, out, flags); } \
    static plan plan_dft_c2r_3d(int n0, int n1, int n2, \
        C *in, R *out, unsigned flags) \
        { return X(plan_dft_c2r_3d)(n0, n1, n2, in, out, flags); } \
\
    static plan plan_guru_dft_r2c(int rank, const iodim *dims, \
        int howmany_rank, const iodim *howmany_dims, \
        R *in, C *out, unsigned flags) \
        { return X(plan_guru_dft_r2c)(rank, dims, \
            howmany_rank, howmany_dims, in, out, flags); } \
    static plan plan_guru_dft_c2r(int rank, const iodim *dims, \
        int howmany_rank, const iodim *howmany_dims, \
        C *in, R *out, unsigned flags) \
        { return X(plan_guru_dft_c2r)(rank, dims, \
            howmany_rank, howmany_dims, in, out, flags); } \
\
    static plan plan_guru_split_dft_r2c( \
        int rank, const iodim *dims, \
        int howmany_rank, const iodim *howmany_dims, \
        R *in, R *ro, R *io, unsigned flags) \
        { return X(plan_guru_split_dft_r2c)(rank, dims, \
            howmany_rank, howmany_dims, in, ro, io, flags); } \
    static plan plan_guru_split_dft_c2r( \
        int rank, const iodim *dims, \
        int howmany_rank, const iodim *howmany_dims, \
        R *ri, R *ii, R *out, unsigned flags) \
        { return X(plan_guru_split_dft_c2r)(rank, dims, \
            howmany_rank, howmany_dims, ri, ii, out, flags); } \
\
    static plan plan_guru64_dft_r2c(int rank, const iodim64 *dims, \
        int howmany_rank, const iodim64 *howmany_dims, \
        R *in, C *out, unsigned flags) \
        { return X(plan_guru64_dft_r2c)(rank, dims, \
            howmany_rank, howmany_dims, in, out, flags); } \
    static plan plan_guru64_dft_c2r(int rank, const iodim64 *dims, \
        int howmany_rank, const iodim64 *howmany_dims, \
        C *in, R *out, unsigned flags) \
        { return X(plan_guru64_dft_c2r)(rank, dims, \
            howmany_rank, howmany_dims, in, out, flags); } \
\
    static plan plan_guru64_split_dft_r2c( \
        int rank, const iodim64 *dims, \
        int howmany_rank, const iodim64 *howmany_dims, \
        R *in, R *ro, R *io, unsigned flags) \
        { return X(plan_guru64_split_dft_r2c)(rank, dims, \
            howmany_rank, howmany_dims, in, ro, io, flags); } \
    static plan plan_guru64_split_dft_c2r( \
        int rank, const iodim64 *dims, \
        int howmany_rank, const iodim64 *howmany_dims, \
        R *ri, R *ii, R *out, unsigned flags) \
        { return X(plan_guru64_split_dft_c2r)(rank, dims, \
            howmany_rank, howmany_dims, ri, ii, out, flags); } \
\
    static void execute_dft_r2c(const plan p, R *in, C *out) \
        { X(execute_dft_r2c)(p, in, out); } \
    static void execute_dft_c2r(const plan p, C *in, R *out) \
        { X(execute_dft_c2r)(p, in, out); } \
\
    static void execute_split_dft_r2c(const plan p, \
        R *in, R *ro, R *io) \
        { X(execute_split_dft_r2c)(p, in, ro, io); } \
    static void execute_split_dft_c2r(const plan p, \
        R *ri, R *ii, R *out) \
        { X(execute_split_dft_c2r)(p, ri, ii, out); } \
\
    static plan plan_many_r2r(int rank, const int *n, int howmany, \
        R *in, const int *inembed, int istride, int idist, \
        R *out, const int *onembed, int ostride, int odist, \
        const r2r_kind *kind, unsigned flags) \
        { return X(plan_many_r2r)(rank, n, howmany, \
            in, inembed, istride, idist, out, onembed, ostride, odist, \
            kind, flags); } \
\
    static plan plan_r2r(int rank, const int *n, R *in, R *out, \
        const r2r_kind *kind, unsigned flags) \
        { return X(plan_r2r)(rank, n, in, out, kind, flags); } \
\
    static plan plan_r2r_1d(int n, R *in, R *out, \
        r2r_kind kind, unsigned flags) \
        { return X(plan_r2r_1d)(n, in, out, kind, flags); } \
    static plan plan_r2r_2d(int n0, int n1, R *in, R *out, \
        r2r_kind kind0, r2r_kind kind1, unsigned flags) \
        { return X(plan_r2r_2d)(n0, n1, in, out, kind0, kind1, flags); } \
    static plan plan_r2r_3d(int n0, int n1, int n2, \
        R *in, R *out, \
        r2r_kind kind0, r2r_kind kind1, r2r_kind kind2, unsigned flags) \
        { return X(plan_r2r_3d)(n0, n1, n2, in, out, \
            kind0, kind1, kind2, flags); } \
\
    static plan plan_guru_r2r(int rank, const iodim *dims, \
        int howmany_rank, const iodim *howmany_dims, \
        R *in, R *out, const r2r_kind *kind, unsigned flags) \
        { return X(plan_guru_r2r)(rank, dims, \
            howmany_rank, howmany_dims, in, out, kind, flags); } \
\
    static plan plan_guru64_r2r(int rank, const iodim64 *dims, \
        int howmany_rank, const iodim64 *howmany_dims, \
        R *in, R *out, const r2r_kind *kind, unsigned flags) \
        { return X(plan_guru64_r2r)(rank, dims, \
            howmany_rank, howmany_dims, in, out, kind, flags); } \
\
    static void execute_r2r(const plan p, R *in, R *out) \
        { X(execute_r2r)(p, in, out); } \
\
    static void destroy_plan(plan p) { X(destroy_plan)(p); } \
    static void forget_wisdom(void)  { X(forget_wisdom)(); } \
    static void cleanup(void)        { X(cleanup)(); } \
\
    static void set_timelimit(double t) \
        { X(set_timelimit)(t); } \
\
    static void plan_with_nthreads(int nthreads) \
        { X(plan_with_nthreads)(nthreads); } \
    static int init_threads(void) \
        { return X(init_threads)(); } \
    static void cleanup_threads(void) \
        { X(cleanup_threads)(); } \
    static void make_planner_thread_safe(void) \
        { X(make_planner_thread_safe)(); } \
\
    static int export_wisdom_to_filename(const char *filename) \
        { return X(export_wisdom_to_filename)(filename); } \
    static void export_wisdom_to_file(FILE *output_file) \
        { X(export_wisdom_to_file)(output_file); } \
    static char *export_wisdom_to_string(void) \
        { return X(export_wisdom_to_string)(); } \
    static void export_wisdom(void (*write_char)(char, void*), void *data) \
        { X(export_wisdom)(write_char, data); } \
\
    static int import_system_wisdom(void) \
        { return X(import_system_wisdom)(); } \
    static int import_wisdom_from_filename(const char *filename) \
        { return X(import_wisdom_from_filename)(filename); } \
    static int import_wisdom_from_file(FILE *input_file) \
        { return X(import_wisdom_from_file)(input_file); } \
    static int import_wisdom_from_string(const char *input_string) \
        { return X(import_wisdom_from_string)(input_string); } \
    static int import_wisdom(int (*read_char)(void*), void *data) \
        { return X(import_wisdom)(read_char, data); } \
\
    static void fprint_plan(const plan p, FILE *output_file) \
        { X(fprint_plan)(p, output_file); } \
    static void print_plan(const plan p) \
        { X(print_plan)(p); } \
    static char *sprint_plan(const plan p) \
        { return X(sprint_plan)(p); } \
\
    static void *malloc(size_t n) \
        { return X(malloc)(n); } \
    static R *alloc_real(size_t n) \
        { return X(alloc_real)(n); } \
    static C *alloc_complex(size_t n) \
        { return X(alloc_complex)(n); } \
    static void free(void *p) \
        { X(free)(p); } \
\
    static void flops(const plan p, \
        double *add, double *mul, double *fmas) \
        { X(flops)(p, add, mul, fmas); } \
    static double estimate_cost(const plan p) \
        { return X(estimate_cost)(p); } \
    static double cost(const plan p) \
        { return X(cost)(p); } \
\
    static int alignment_of(R *p) \
        { return X(alignment_of)(p); } \
    static const char *version(void) \
        { return X(version); } \
    static const char *cc(void) \
        { return X(cc); } \
    static const char *codelet_optim(void) \
        { return X(codelet_optim); } \
}; // struct fftw


template<typename T>
struct fftw {};

// use the macro to define the differently typed APIs
FFTW3CXX_DEFINE_API(FFTW_MANGLE_DOUBLE, double, fftw_complex)
FFTW3CXX_DEFINE_API(FFTW_MANGLE_FLOAT, float, fftwf_complex)
FFTW3CXX_DEFINE_API(FFTW_MANGLE_LONG_DOUBLE, long double, fftwl_complex)

// Quad float is a gcc extension and would need some attention.
// FFTW3CXX_DEFINE_API(FFTW_MANGLE_QUAD, __float128, fftwq_complex)


//---
// A plan class using RAII and with inline methods
// that forward to the above fftw3cxx::fftw<T>:: API
// Supports both fftw's complex type and C++ std::complex.
//---

template<typename T>
class plan {
  public:
    typedef plan<T> self_type;
    typedef T R;
    typedef typename fftw<T>::complex C;
    typedef std::complex<T> SC;
    typedef R real;
    typedef C complex;
    typedef SC std_complex;
    typedef typename fftw<T>::iodim iodim;
    typedef typename fftw<T>::iodim64 iodim64;
    typedef typename fftw<T>::r2r_kind r2r_kind;

    plan(void): paux(0) {}
    plan(typename fftw<T>::plan p): paux(new aux(p)) {}
    plan(const self_type &other): paux(other.paux) { inc(); }
   ~plan(void) { dec(); }
    void swap(self_type &other)
        { aux *tmp = other.paux; other.paux = paux; paux = tmp; }
    self_type& operator=(self_type other)
        { swap(other); return *this; }
    void clear(void) { self_type t; *this = t; }
    bool empty(void) const { return paux==0; }

    // What follows is most of the fftw3 API
    // in the same order it appears in fftw3.h
    // Plan creators are static named constructors and
    // plan executors are methods

    void execute(void) const
        { fftw<T>::execute(p()); }

    static self_type plan_dft(int rank, const int *n,
        C *in, C *out, int sign, unsigned flags)
        { return fftw<T>::plan_dft(rank, n, in, out, sign, flags); }
    static self_type plan_dft(int rank, const int *n,
        SC *in, SC *out, int sign, unsigned flags)
        { return fftw<T>::plan_dft(rank, n, (C*)in, (C*)out, sign, flags); }

    static self_type plan_dft_1d(int n,
        C *in, C *out, int sign, unsigned flags)
        { return fftw<T>::plan_dft_1d(n, in, out, sign, flags); }
    static self_type plan_dft_1d(int n,
        SC *in, SC *out, int sign, unsigned flags)
        { return fftw<T>::plan_dft_1d(n, (C*)in, (C*)out, sign, flags); }
    static self_type plan_dft_2d(int n0, int n1,
        C *in, C *out, int sign, unsigned flags)
        { return fftw<T>::plan_dft_2d(n0, n1, in, out, sign, flags); }
    static self_type plan_dft_2d(int n0, int n1,
        SC *in, SC *out, int sign, unsigned flags)
        { return fftw<T>::plan_dft_2d(n0, n1, (C*)in, (C*)out, sign, flags); }
    static self_type plan_dft_3d(int n0, int n1, int n2,
        C *in, C *out, int sign, unsigned flags)
        { return fftw<T>::plan_dft_3d(n0, n1, n2, in, out, sign, flags); }
    static self_type plan_dft_3d(int n0, int n1, int n2,
        SC *in, SC *out, int sign, unsigned flags)
        { return fftw<T>::plan_dft_3d(n0, n1, n2, (C*)in, (C*)out, sign, flags); }

    static self_type plan_many_dft(int rank, const int *n, int howmany,
        C *in, const int *inembed, int istride, int idist,
        C *out, const int *onembed, int ostride, int odist,
        int sign, unsigned flags)
        { return fftw<T>::plan_many_dft(rank, n, howmany,
            in, inembed, istride, idist,
            out, onembed, ostride, odist, sign, flags); }
    static self_type plan_many_dft(int rank, const int *n, int howmany,
        SC *in, const int *inembed, int istride, int idist,
        SC *out, const int *onembed, int ostride, int odist,
        int sign, unsigned flags)
        { return fftw<T>::plan_many_dft(rank, n, howmany,
            (C*)in, inembed, istride, idist,
            (C*)out, onembed, ostride, odist, sign, flags); }

    static self_type plan_guru_dft(int rank, const iodim *dims,
        int howmany_rank, const iodim *howmany_dims,
        C *in, C *out, int sign, unsigned flags)
        { return fftw<T>::plan_guru_dft(rank, dims,
            howmany_rank, howmany_dims, in, out, sign, flags); }
    static self_type plan_guru_dft(int rank, const iodim *dims,
        int howmany_rank, const iodim *howmany_dims,
        SC *in, SC *out, int sign, unsigned flags)
        { return fftw<T>::plan_guru_dft(rank, dims,
            howmany_rank, howmany_dims, (C*)in, (C*)out, sign, flags); }
    static self_type plan_guru_split_dft(int rank, const iodim *dims,
        int howmany_rank, const iodim *howmany_dims,
        R *ri, R *ii, R *ro, R *io, unsigned flags)
        { return fftw<T>::plan_guru_split_dft(rank, dims,
            howmany_rank, howmany_dims, ri, ii, ro, io, flags); }

    static self_type plan_guru64_dft(int rank, const iodim64 *dims,
        int howmany_rank, const iodim64 *howmany_dims,
        C *in, C *out, int sign, unsigned flags)
        { return fftw<T>::plan_guru64_dft(rank, dims,
            howmany_rank, howmany_dims, in, out, sign, flags); }
    static self_type plan_guru64_dft(int rank, const iodim64 *dims,
        int howmany_rank, const iodim64 *howmany_dims,
        SC *in, SC *out, int sign, unsigned flags)
        { return fftw<T>::plan_guru64_dft(rank, dims,
            howmany_rank, howmany_dims, (C*)in, (C*)out, sign, flags); }
    static self_type plan_guru64_split_dft(int rank, const iodim64 *dims,
        int howmany_rank, const iodim64 *howmany_dims,
        R *ri, R *ii, R *ro, R *io, unsigned flags)
        { return fftw<T>::plan_guru64_split_dft(rank, dims,
            howmany_rank, howmany_dims, ri, ii, ro, io, flags); }

    void execute_dft(C *in, C *out) const
        { fftw<T>::execute_dft(p(), in, out); }
    void execute_dft(SC *in, SC *out) const
        { fftw<T>::execute_dft(p(), (C*)in, (C*)out); }
    void execute_split_dft(R *ri, R *ii, R *ro, R *io) const
        { fftw<T>::execute_split_dft(p(), ri, ii, ro, io); }

    static self_type plan_many_dft_r2c(int rank, const int *n, int howmany,
        R *in, const int *inembed, int istride, int idist,
        C *out, const int *onembed, int ostride, int odist,
        unsigned flags)
        { return fftw<T>::plan_many_dft_r2c(rank, n, howmany,
            in, inembed, istride, idist,
            out, onembed, ostride, odist, flags); }
    static self_type plan_many_dft_r2c(int rank, const int *n, int howmany,
        R *in, const int *inembed, int istride, int idist,
        SC *out, const int *onembed, int ostride, int odist,
        unsigned flags)
        { return fftw<T>::plan_many_dft_r2c(rank, n, howmany,
            in, inembed, istride, idist,
            (C*)out, onembed, ostride, odist, flags); }

    static self_type plan_dft_r2c(int rank, const int *n,
        R *in, C *out, unsigned flags)
        { return fftw<T>::plan_dft_r2c(rank, n, in, out, flags); }
    static self_type plan_dft_r2c(int rank, const int *n,
        R *in, SC *out, unsigned flags)
        { return fftw<T>::plan_dft_r2c(rank, n, in, (C*)out, flags); }

    static self_type plan_dft_r2c_1d(int n, R *in, C *out,
        unsigned flags)
        { return fftw<T>::plan_dft_r2c_1d(n, in, out, flags); }
    static self_type plan_dft_r2c_1d(int n, R *in, SC *out,
        unsigned flags)
        { return fftw<T>::plan_dft_r2c_1d(n, in, (C*)out, flags); }
    static self_type plan_dft_r2c_2d(int n0, int n1,
        R *in, C *out, unsigned flags)
        { return fftw<T>::plan_dft_r2c_2d(n0, n1, in, out, flags); }
    static self_type plan_dft_r2c_2d(int n0, int n1,
        R *in, SC *out, unsigned flags)
        { return fftw<T>::plan_dft_r2c_2d(n0, n1, in, (C*)out, flags); }
    static self_type plan_dft_r2c_3d(int n0, int n1, int n2,
        R *in, C *out, unsigned flags)
        { return fftw<T>::plan_dft_r2c_3d(n0, n1, n2, in, out, flags); }
    static self_type plan_dft_r2c_3d(int n0, int n1, int n2,
        R *in, SC *out, unsigned flags)
        { return fftw<T>::plan_dft_r2c_3d(n0, n1, n2, in, (C*)out, flags); }

    static self_type plan_many_dft_c2r(int rank, const int *n, int howmany,
        C *in, const int *inembed, int istride, int idist,
        R *out, const int *onembed, int ostride, int odist,
        unsigned flags)
        { return fftw<T>::plan_many_dft_c2r(rank, n, howmany,
            in, inembed, istride, idist,
            out, onembed, ostride, odist, flags); }
    static self_type plan_many_dft_c2r(int rank, const int *n, int howmany,
        SC *in, const int *inembed, int istride, int idist,
        R *out, const int *onembed, int ostride, int odist,
        unsigned flags)
        { return fftw<T>::plan_many_dft_c2r(rank, n, howmany,
            (C*)in, inembed, istride, idist,
            out, onembed, ostride, odist, flags); }

    static self_type plan_dft_c2r(int rank, const int *n,
        C *in, R *out, unsigned flags)
        { return fftw<T>::plan_dft_c2r(rank, n, in, out, flags); }
    static self_type plan_dft_c2r(int rank, const int *n,
        SC *in, R *out, unsigned flags)
        { return fftw<T>::plan_dft_c2r(rank, n, (C*)in, out, flags); }

    static self_type plan_dft_c2r_1d(int n, C *in, R *out, unsigned flags)
        { return fftw<T>::plan_dft_c2r_1d(n, in, out, flags); }
    static self_type plan_dft_c2r_1d(int n, SC *in, R *out, unsigned flags)
        { return fftw<T>::plan_dft_c2r_1d(n, (C*)in, out, flags); }
    static self_type plan_dft_c2r_2d(int n0, int n1,
        C *in, R *out, unsigned flags)
        { return fftw<T>::plan_dft_c2r_2d(n0, n1, in, out, flags); }
    static self_type plan_dft_c2r_2d(int n0, int n1,
        SC *in, R *out, unsigned flags)
        { return fftw<T>::plan_dft_c2r_2d(n0, n1, (C*)in, out, flags); }
    static self_type plan_dft_c2r_3d(int n0, int n1, int n2,
        C *in, R *out, unsigned flags)
        { return fftw<T>::plan_dft_c2r_3d(n0, n1, n2, in, out, flags); }
    static self_type plan_dft_c2r_3d(int n0, int n1, int n2,
        SC *in, R *out, unsigned flags)
        { return fftw<T>::plan_dft_c2r_3d(n0, n1, n2, (C*)in, out, flags); }

    static self_type plan_guru_dft_r2c(int rank, const iodim *dims,
        int howmany_rank, const iodim *howmany_dims,
        R *in, C *out, unsigned flags)
        { return fftw<T>::plan_guru_dft_r2c(rank, dims,
            howmany_rank, howmany_dims, in, out, flags); }
    static self_type plan_guru_dft_r2c(int rank, const iodim *dims,
        int howmany_rank, const iodim *howmany_dims,
        R *in, SC *out, unsigned flags)
        { return fftw<T>::plan_guru_dft_r2c(rank, dims,
            howmany_rank, howmany_dims, in, (C*)out, flags); }
    static self_type plan_guru_dft_c2r(int rank, const iodim *dims,
        int howmany_rank, const iodim *howmany_dims,
        C *in, R *out, unsigned flags)
        { return fftw<T>::plan_guru_dft_c2r(rank, dims,
            howmany_rank, howmany_dims, in, out, flags); }
    static self_type plan_guru_dft_c2r(int rank, const iodim *dims,
        int howmany_rank, const iodim *howmany_dims,
        SC *in, R *out, unsigned flags)
        { return fftw<T>::plan_guru_dft_c2r(rank, dims,
            howmany_rank, howmany_dims, (C*)in, out, flags); }

    static self_type plan_guru_split_dft_r2c(
        int rank, const iodim *dims,
        int howmany_rank, const iodim *howmany_dims,
        R *in, R *ro, R *io, unsigned flags)
        { return fftw<T>::plan_guru_split_dft_r2c(rank, dims,
            howmany_rank, howmany_dims, in, ro, io, flags); }
    static self_type plan_guru_split_dft_c2r(
        int rank, const iodim *dims,
        int howmany_rank, const iodim *howmany_dims,
        R *ri, R *ii, R *out, unsigned flags)
        { return fftw<T>::plan_guru_split_dft_c2r(rank, dims,
            howmany_rank, howmany_dims, ri, ii, out, flags); }

    static self_type plan_guru64_dft_r2c(int rank, const iodim64 *dims,
        int howmany_rank, const iodim64 *howmany_dims,
        R *in, C *out, unsigned flags)
        { return fftw<T>::plan_guru64_dft_r2c(rank, dims,
            howmany_rank, howmany_dims, in, out, flags); }
    static self_type plan_guru64_dft_r2c(int rank, const iodim64 *dims,
        int howmany_rank, const iodim64 *howmany_dims,
        R *in, SC *out, unsigned flags)
        { return fftw<T>::plan_guru64_dft_r2c(rank, dims,
            howmany_rank, howmany_dims, in, (C*)out, flags); }
    static self_type plan_guru64_dft_c2r(int rank, const iodim64 *dims,
        int howmany_rank, const iodim64 *howmany_dims,
        C *in, R *out, unsigned flags)
        { return fftw<T>::plan_guru64_dft_c2r(rank, dims,
            howmany_rank, howmany_dims, in, out, flags); }
    static self_type plan_guru64_dft_c2r(int rank, const iodim64 *dims,
        int howmany_rank, const iodim64 *howmany_dims,
        SC *in, R *out, unsigned flags)
        { return fftw<T>::plan_guru64_dft_c2r(rank, dims,
            howmany_rank, howmany_dims, (C*)in, out, flags); }

    static self_type plan_guru64_split_dft_r2c(
        int rank, const iodim64 *dims,
        int howmany_rank, const iodim64 *howmany_dims,
        R *in, R *ro, R *io, unsigned flags)
        { return fftw<T>::plan_guru64_split_dft_r2c(rank, dims,
            howmany_rank, howmany_dims, in, ro, io, flags); }
    static self_type plan_guru64_split_dft_c2r(
        int rank, const iodim64 *dims,
        int howmany_rank, const iodim64 *howmany_dims,
        R *ri, R *ii, R *out, unsigned flags)
        { return fftw<T>::plan_guru64_split_dft_c2r(rank, dims,
            howmany_rank, howmany_dims, ri, ii, out, flags); }

    void execute_dft_r2c(R *in, C *out) const
        { fftw<T>::execute_dft_r2c(p(), in, out); }
    void execute_dft_r2c(R *in, SC *out) const
        { fftw<T>::execute_dft_r2c(p(), in, (C*)out); }
    void execute_dft_c2r(C *in, R *out) const
        { fftw<T>::execute_dft_c2r(p(), in, out); }
    void execute_dft_c2r(SC *in, R *out) const
        { fftw<T>::execute_dft_c2r(p(), (C*)in, out); }

    void execute_split_dft_r2c(R *in, R *ro, R *io) const
        { fftw<T>::execute_split_dft_r2c(p(), in, ro, io); }
    void execute_split_dft_c2r(R *ri, R *ii, R *out) const
        { fftw<T>::execute_split_dft_c2r(p(), ri, ii, out); }

    static self_type plan_many_r2r(int rank, const int *n, int howmany,
        R *in, const int *inembed, int istride, int idist,
        R *out, const int *onembed, int ostride, int odist,
        const r2r_kind *kind, unsigned flags)
        { return fftw<T>::plan_many_r2r(rank, n, howmany,
            in, inembed, istride, idist, out, onembed, ostride, odist,
            kind, flags); }

    static self_type plan_r2r(int rank, const int *n, R *in, R *out,
        const r2r_kind *kind, unsigned flags)
        { return fftw<T>::plan_r2r(rank, n, in, out, kind, flags); }

    static self_type plan_r2r_1d(int n, R *in, R *out,
        r2r_kind kind, unsigned flags)
        { return fftw<T>::plan_r2r_1d(n, in, out, kind, flags); }
    static self_type plan_r2r_2d(int n0, int n1, R *in, R *out,
        r2r_kind kind0, r2r_kind kind1, unsigned flags)
        { return fftw<T>::plan_r2r_2d(n0, n1, in, out, kind0, kind1, flags); }
    static self_type plan_r2r_3d(int n0, int n1, int n2,
        R *in, R *out,
        r2r_kind kind0, r2r_kind kind1, r2r_kind kind2, unsigned flags)
        { return fftw<T>::plan_r2r_3d(n0, n1, n2, in, out,
            kind0, kind1, kind2, flags); }

    static self_type plan_guru_r2r(int rank, const iodim *dims,
        int howmany_rank, const iodim *howmany_dims,
        R *in, R *out, const r2r_kind *kind, unsigned flags)
        { return fftw<T>::plan_guru_r2r(rank, dims,
            howmany_rank, howmany_dims, in, out, kind, flags); }

    static self_type plan_guru64_r2r(int rank, const iodim64 *dims,
        int howmany_rank, const iodim64 *howmany_dims,
        R *in, R *out, const r2r_kind *kind, unsigned flags)
        { return fftw<T>::plan_guru64_r2r(rank, dims,
            howmany_rank, howmany_dims, in, out, kind, flags); }

    void execute_r2r(R *in, R *out) const
        { fftw<T>::execute_r2r(p(), in, out); }

    // This ends following the order of fftw3.h

    // additional plan methods
    void fprint(FILE *output_file) const
        { fftw<T>::fprint_plan(p(), output_file); }
    void print(void) const
        { fftw<T>::print_plan(p()); }
    char *sprint(void) const
        { return fftw<T>::sprint_plan(p()); }
    void flops(double *add, double *mul, double *fmas) const
        { fftw<T>::flops(p(), add, mul, fmas); }
    double estimate_cost(void) const
        { return fftw<T>::estimate_cost(p()); }
    double cost(void) const
        { return fftw<T>::cost(p()); }

  private:
    class aux {
        typename fftw<T>::plan p;
        unsigned refcnt;
        aux(typename fftw<T>::plan p_): p(p_), refcnt(1) {}
       ~aux(void) { fftw<T>::destroy_plan(p); }
        void inc(void) { ++refcnt; }
        unsigned dec(void) { return --refcnt; }
        friend class plan<T>;
    };
    aux *paux;
    void inc(void) { if (paux) paux->inc(); }
    void dec(void) { if (paux && !paux->dec()) { delete paux; paux = 0; } }
    typename fftw<T>::plan p(void) const {
        if (!paux) throw std::runtime_error("plan is not initialized");
        return paux->p;
    }
};

//---
// Define template functions to put the remaining (non-plan) FFTW API
// directly into the fftw3cxx namespace.
// e.g. fftw3cxx::fftw<double>::forget_wisdom is fftw3cxx::forget_wisdom<double>
//---

// The remainder appears in the same order as fftw3.h

template<typename T>
inline void forget_wisdom(void)
    { fftw<T>::forget_wisdom(); }
template<typename T>
inline void cleanup(void)
    { fftw<T>::cleanup(); }

template<typename T>
inline void set_timelimit(double t)
    { fftw<T>::set_timelimit(t); }

template<typename T>
inline void plan_with_nthreads(int nthreads)
    { fftw<T>::plan_with_nthreads(nthreads); }
template<typename T>
inline int init_threads(void)
    { return fftw<T>::init_threads(); }
template<typename T>
inline void cleanup_threads(void)
    { fftw<T>::cleanup_threads(); }
template<typename T>
inline void make_planner_thread_safe(void)
    { fftw<T>::make_planner_thread_safe(); }

template<typename T>
inline int export_wisdom_to_filename(const char *filename)
    { return fftw<T>::export_wisdom_to_filename(filename); }
template<typename T>
inline void export_wisdom_to_file(FILE *output_file)
    { fftw<T>::export_wisdom_to_file(output_file); }
template<typename T>
inline char *export_wisdom_to_string(void)
    { return fftw<T>::export_wisdom_to_string(); }
template<typename T>
inline void export_wisdom(void (*write_char)(char, void*), void *data)
    { fftw<T>::export_wisdom(write_char, data); }

template<typename T>
inline int import_system_wisdom(void)
    { return fftw<T>::import_system_wisdom(); }
template<typename T>
inline int import_wisdom_from_filename(const char *filename)
    { return fftw<T>::import_wisdom_from_filename(filename); }
template<typename T>
inline int import_wisdom_from_file(FILE *input_file)
    { return fftw<T>::import_wisdom_from_file(input_file); }
template<typename T>
inline int import_wisdom_from_string(const char *input_string)
    { return fftw<T>::import_wisdom_from_string(input_string); }
template<typename T>
inline int import_wisdom(int (*read_char)(void*), void *data)
    { return fftw<T>::import_wisdom(read_char, data); }

template<typename T>
inline void *malloc(size_t n)
    { return fftw<T>::malloc(n); }
template<typename T>
inline T *alloc_real(size_t n)
    { return fftw<T>::alloc_real(n); }
template<typename T>
inline typename fftw<T>::complex *alloc_complex(size_t n)
    { return fftw<T>::alloc_complex(n); }
template<typename T>
inline void free(void *p)
    { fftw<T>::free(p); }

template<typename T>
inline int alignment_of(T *p)
    { return fftw<T>::alignment_of(p); }
template<typename T>
inline const char *version(void)
    { return fftw<T>::version; }
template<typename T>
inline const char *cc(void)
    { return fftw<T>::cc; }
template<typename T>
inline const char *codelet_optim(void)
    { return fftw<T>::codelet_optim; }

} // namespace fftw3cxx
} // namespace isce
#endif
