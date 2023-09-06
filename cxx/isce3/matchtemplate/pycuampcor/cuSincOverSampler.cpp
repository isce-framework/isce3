/**
 * @file cuSincOverSampler.cu
 * @brief Implementation for cuSinOversampler class
 *
 */

// my declaration
#include "cuSincOverSampler.h"

// dependencies
#include "cuArrays.h"
#include "cuAmpcorUtil.h"

namespace isce3::matchtemplate::pycuampcor {

/**
 * cuSincOverSamplerR2R constructor
 * @param i_covs oversampling factor
 */
cuSincOverSamplerR2R::cuSincOverSamplerR2R(const int i_covs_)
 : i_covs(i_covs_)
{
    i_intplength = int(r_relfiltlen/r_beta+0.5f);
    i_filtercoef = i_intplength*i_decfactor;
    r_filter = (float*) malloc((i_filtercoef+1)*sizeof(float));
    cuSetupSincKernel();
}

/// destructor
cuSincOverSamplerR2R::~cuSincOverSamplerR2R()
{
    free(r_filter);
}

// cuda kernel for cuSetupSincKernel
void cuSetupSincKernel_kernel(float *r_filter_, const int i_filtercoef_,
    const float r_soff_, const float r_wgthgt_, const int i_weight_,
    const float r_soff_inverse_, const float r_beta_, const float r_decfactor_inverse_, int i)
{
    if(i > i_filtercoef_) return;
    float r_wa = i - r_soff_;
    float r_wgt = (1.0f - r_wgthgt_) + r_wgthgt_*cos(M_PI*r_wa*r_soff_inverse_);
    float r_s = r_wa*r_beta_*r_decfactor_inverse_*M_PI;
    float r_fct;
    if(r_s != 0.0f) {
        r_fct = sin(r_s)/r_s;
    }
    else {
        r_fct = 1.0f;
    }
    if(i_weight_ == 1) {
        r_filter_[i] = r_fct*r_wgt;
    }
    else {
        r_filter_[i] = r_fct;
    }
}

/**
 * Set up the sinc interpolation kernel (coefficient)
 */
void cuSincOverSamplerR2R::cuSetupSincKernel()
{
    // compute some commonly used constants at first
    float r_wgthgt =  (1.0f - r_pedestal)/2.0f;
    float r_soff = (i_filtercoef-1.0f)/2.0f;
    float r_soff_inverse = 1.0f/r_soff;
    float r_decfactor_inverse = 1.0f/i_decfactor;

    for (int i = 0; i <= i_filtercoef; i++) {

        cuSetupSincKernel_kernel(
            r_filter, i_filtercoef, r_soff, r_wgthgt, i_weight,
            r_soff_inverse, r_beta, r_decfactor_inverse, i);
    }
}


// cuda kernel for cuSincOverSamplerR2R::execute
void cuSincInterpolation_kernel(const int nImages,
    const float * imagesIn, const int inNX, const int inNY,
    float * imagesOut, const int outNX, const int outNY,
    int2 *centerShift, int factor,
    const float * r_filter_, const int i_covs_, const int i_decfactor_, const int i_intplength_,
    const int i_startX, const int i_startY, const int i_int_size,
    int idxImage, int idxX, int idxY)
{
    // decide the center shift
    int2 shift = centerShift[idxImage];
    // determine the output pixel indices
    int outx = idxX + i_startX + shift.x*factor;
    if (outx >= outNX) outx-=outNX;
    int outy = idxY + i_startY +  shift.y*factor;
    if (outy >= outNY) outy-=outNY;
    // flattened to 1d
    int idxOut = idxImage*outNX*outNY + outx*outNY + outy;

    // index in input grids
    float r_xout = (float)outx/i_covs_;
     // integer part
    int i_xout = int(r_xout);
    // factional part
    float r_xfrac = r_xout - i_xout;
    // fractional part in terms of the interpolation kernel grids
    int i_xfrac = int(r_xfrac*i_decfactor_);

    // same procedure for y
    float r_yout = (float)outy/i_covs_;
    int i_yout = int(r_yout);
    float r_yfrac = r_yout - i_yout;
    int i_yfrac = int(r_yfrac*i_decfactor_);

    // temp variables
    float intpData = 0.0f; // interpolated value
    float r_sincwgt = 0.0f; // total filter weight
    float r_sinc_coef; // filter weight

    // iterate over lines of input image
    // i=0 -> -i_intplength/2
    for(int i=0; i < i_intplength_; i++) {
        // find the corresponding pixel in input(unsampled) image

        int inx = i_xout - i + i_intplength_/2;

        if(inx < 0) inx+= inNX;
        if(inx >= inNX) inx-= inNY;

        float r_xsinc_coef = r_filter_[i*i_decfactor_+i_xfrac];

        for(int j=0; j< i_intplength_; j++) {
            // find the corresponding pixel in input(unsampled) image
            int iny = i_yout - j + i_intplength_/2;
            if(iny < 0) iny += inNY;
            if(iny >= inNY) iny -= inNY;

            float r_ysinc_coef = r_filter_[j*i_decfactor_+i_yfrac];
            // multiply the factors from xy
            r_sinc_coef = r_xsinc_coef*r_ysinc_coef;
            // add to total sinc weight
            r_sincwgt += r_sinc_coef;
            // multiply by the original signal and add to results
            intpData += imagesIn[idxImage*inNX*inNY+inx*inNY+iny]*r_sinc_coef;

        }
    }
    imagesOut[idxOut] = intpData/r_sincwgt;
}

/**
 * Execute sinc interpolation
 * @param[in] imagesIn input images
 * @param[out] imagesOut output images
 * @param[in] centerShift the shift of interpolation center
 * @param[in] rawOversamplingFactor the multiplier of the centerShift
 * @note rawOversamplingFactor is for the centerShift, not the signal oversampling factor
 */

void cuSincOverSamplerR2R::execute(cuArrays<float> *imagesIn, cuArrays<float> *imagesOut,
    cuArrays<int2> *centerShift, int rawOversamplingFactor)
{
    const int nImages = imagesIn->count;
    const int inNX = imagesIn->height;
    const int inNY = imagesIn->width;
    const int outNX = imagesOut->height;
    const int outNY = imagesOut->width;

    // only compute the overampled signals within a window
    const int i_int_range = i_sincwindow * i_covs;
    // set the start pixel, will be shifted by centerShift*oversamplingFactor (from raw image)
    const int i_int_startX = outNX/2 - i_int_range;
    const int i_int_startY = outNY/2 - i_int_range;
    const int i_int_size = 2*i_int_range + 1;
    // preset all pixels in out image to 0
    imagesOut->setZero();

    for (int idxImage = 0; idxImage < nImages; idxImage++) {
        for (int idxX = 0; idxX < i_int_size; idxX++) {
            for (int idxY = 0; idxY < i_int_size; idxY++) {
                cuSincInterpolation_kernel(nImages,
                    imagesIn->devData, inNX, inNY,
                    imagesOut->devData, outNX, outNY,
                    centerShift->devData, rawOversamplingFactor,
                    r_filter, i_covs, i_decfactor, i_intplength, i_int_startX, i_int_startY, i_int_size,
                    idxImage, idxX, idxY);
            }
        }
    }
}

} // namespace
