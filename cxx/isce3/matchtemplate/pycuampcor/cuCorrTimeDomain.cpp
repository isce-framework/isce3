#include "cuAmpcorUtil.h"

namespace isce3::matchtemplate::pycuampcor {


/**
 * Straightforward 2D image convolution
 * between template (reference) and search (secondary) images.
 */
void cuCorrTimeDomain(cuArrays<float> *templates,
               cuArrays<float> *images,
               cuArrays<float> *results)
{
    for (int i = 0; i < results->count; i++) {
        // For each image, get the reference and secondary memory regions,
        // as well as the output region. (These are 2D images
        // but are accessed here using flat array indexing.)
        const float* image = &images->devData[i * images->size];
        const float* templ = &templates->devData[i * templates->size];
        float* result = &results->devData[i * results->size];

        for (int y = 0; y < results->height; y++) {
            for (int x = 0; x < results->width; x++) {
                // This is the value of this pixel in the output convolution,
                // that will be accumulated for each pixel in the search image.
                float pixel = 0;
                for (int y0 = 0; y0 < templates->height; y0++) {
                    for (int x0 = 0; x0 < templates->width; x0++) {
                        const float template_pixel = templ[y0 * templates->width + x0];
                        const float search_pixel = image[(y+y0) * images->width + (x+x0)];
                        pixel += template_pixel * search_pixel;
                    }
                }
                result[y * results->width + x] = pixel;
            }
        }
    }
}

} // namespace
