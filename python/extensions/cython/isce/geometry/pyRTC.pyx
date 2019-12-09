import numbers
from cython.operator cimport dereference as deref
from RTC cimport facetRTC, rtcInputRadiometry, rtcOutputMode

rtc_input_radiometry_dict = { 'BETA_NAUGHT': rtcInputRadiometry.BETA_NAUGHT,
                              'SIGMA_NAUGHT': rtcInputRadiometry.SIGMA_NAUGHT}
        
rtc_output_mode_dict = {'GAMMA_NAUGHT_AREA': rtcOutputMode.GAMMA_NAUGHT_AREA,
                        'GAMMA_NAUGHT_DIVISOR': rtcOutputMode.GAMMA_NAUGHT_DIVISOR}

def pyRTC_impl(pyProduct prod, pyRaster in_raster, pyRaster out_raster,
               char frequency=b'A',
               input_radiometry=None,
               output_mode=None):

    # input radiometry
    rtc_input_radiometry = None
    if input_radiometry is None:
        input_radiometry_key='SIGMA_NAUGHT'
    elif isinstance(input_radiometry, numbers.Number):
        rtc_input_radiometry = input_radiometry
    else:
        input_radiometry_key = input_radiometry.upper().replace('-', '_')
    if rtc_input_radiometry is None:
        rtc_input_radiometry = rtc_input_radiometry_dict[input_radiometry_key]
 
    # output mode
    rtc_output_mode = None
    if output_mode is None:
        output_mode_key='GAMMA_NAUGHT_AREA'
    elif isinstance(output_mode, numbers.Number):
        rtc_output_mode = output_mode
    else:
        output_mode_key = output_mode.upper().replace('-', '_')
    if rtc_output_mode is None:
        rtc_output_mode = rtc_output_mode_dict[output_mode_key]

    facetRTC(deref(prod.c_product),
             deref(in_raster.c_raster),
             deref(out_raster.c_raster),
             frequency,
             rtc_input_radiometry,
             rtc_output_mode)

# Wrapper to support output as filename or pyRaster
def pyRTC(pyProduct prod, pyRaster in_raster, out_raster, 
          char frequency=b'A',
          input_radiometry=None,
          output_mode=None):

    # Type-check output raster
    if type(out_raster) != pyRaster:
        # Create output raster if filename is given
        if type(out_raster) == str:
            filename = out_raster
            out_raster = pyRaster(filename, access=1, width=in_raster.width,
                                                      length=in_raster.length)
        else:
            raise TypeError("must pass pyRaster or filename to pyRTC")
    else:
        # Enforce output raster is writable
        if out_raster.access != 1:
            raise ValueError("output raster must be writable")

    pyRTC_impl(prod, in_raster, out_raster, frequency=frequency,
               input_radiometry=input_radiometry, output_mode=output_mode)