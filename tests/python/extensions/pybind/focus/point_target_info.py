#!/usr/bin/env python3
"""
Analyze a point target in a complex*8 file.
"""
import sys
import numpy as np

desc = __doc__


def get_chip (x, i, j, nchip=64):
    i = int(i)
    j = int(j)
    chip = np.zeros ((nchip,nchip), dtype=x.dtype)
    nchip2 = nchip // 2
    i0 = i - nchip2 + 1
    i1 = i0 + nchip
    j0 = j - nchip2 + 1
    j1 = j0 + nchip
    #FIXME handle edge cases by zero-padding
    chip[:,:] = x[i0:i1, j0:j1]
    return i0, j0, chip


def estimate_frequency (z):
    cx = np.sum (z[:,1:] * z[:,:-1].conj())
    cy = np.sum (z[1:,:] * z[:-1,:].conj())
    return np.angle ([cx, cy])


def shift_frequency (z, fx, fy):
    x, y = np.meshgrid (list(range(z.shape[1])), list(range(z.shape[0])))
    z *= np.exp (1j * fx * x)
    z *= np.exp (1j * fy * y)
    return z


def oversample (x, nov, baseband=False, return_slopes=False):
    m, n = x.shape
    assert (m==n)

    if (not baseband):
        # shift the data to baseband
        fx, fy = estimate_frequency (x)
        x = shift_frequency (x, -fx, -fy)

    X = np.fft.fft2 (x)
    # Zero-pad high frequencies in the spectrum.
    Y = np.zeros ((n*nov, n*nov), dtype=X.dtype)
    n2 = n // 2
    Y[:n2, :n2] = X[:n2, :n2]
    Y[-n2:, -n2:] = X[-n2:, -n2:]
    Y[:n2, -n2:] = X[:n2, -n2:]
    Y[-n2:, :n2] = X[-n2:, :n2]
    # Split Nyquist bins symmetrically.
    assert n % 2 == 0
    Y[:n2,n2] = Y[:n2,-n2] = 0.5 * X[:n2,n2]
    Y[-n2:,n2] = Y[-n2:,-n2] = 0.5 * X[-n2:,n2]
    Y[n2,:n2] = Y[-n2,:n2] = 0.5 * X[n2,:n2]
    Y[n2,-n2:] = Y[-n2,-n2:] = 0.5 * X[n2,-n2:]
    Y[n2,n2] = Y[n2,-n2] = Y[-n2,n2] = Y[-n2,-n2] = 0.25 * X[n2,n2]
    # Back to time domain.
    y = np.fft.ifft2 (Y)
    # NOTE account for scaling of different-sized DFTs.
    y *= nov**2

    if (not baseband):
        # put the phase back on
        y = shift_frequency (y, fx/nov, fy/nov)

    y = np.asarray (y, dtype=x.dtype)
    if return_slopes:
        return (y, fx, fy)
    return y


def estimate_resolution (x, dt=1.0):
    # Find the peak.
    y = abs (x)**2
    i = np.argmax (y)
    # Construct a function with zeros at the -3dB points.
    u = y - 0.5 * y[i]
    # Make sure the interval contains a peak.  If not, return interval width.
    if ((u[0] >= 0.0) or (u[-1] >= 0.0)):
        print('Warning: Interval does not contain a well-defined peak.',
              file=sys.stderr)
        return dt * len (x)
    # Take its absolute value so can search for minima instead of intersections.
    z = abs (u)
    # Find the points on each side of the peak.
    left = z[:i]
    ileft = np.argmin (left)
    right = z[i:]
    iright = i + np.argmin (right)
    # Return the distance between -3dB crossings, scaled by the sample spacing.
    return dt * (iright - ileft)

def find_null_to_null(matched_output, num_nulls_main, num_samples_null, main_peak_idx):
    """Compute mainlobe null locations as sample index for ISLR and PSLR.

    Parameters:
    -----------
    matched_output: complex array, Range or Azimuth cuts
    num_nulls_main: int, num_nulls_main = 2 if mainlobe includes first sidelobes
    num_samples_null: float, default is Fs/bandwidth, number of samples from null to null

    Returns:
    --------
    nullLeftIdx: null location left of mainlobe peak
    nullRightIdx: null location right of mainlobe peak
    """
    
    #Search at least 1 sample beyond expected null
    num_samples_search = num_samples_null + 2
    
    if (num_nulls_main == 1):
        first_peak_left_idx = main_peak_idx
        first_peak_right_idx = main_peak_idx
        
        search_samples_left_stop = first_peak_left_idx - num_samples_search
        search_samples_right_stop = first_peak_right_idx + num_samples_search
               
        samples_left = matched_output[first_peak_left_idx : search_samples_left_stop : -1]
        samples_right = matched_output[first_peak_right_idx : search_samples_right_stop] 
    elif (num_nulls_main == 2):
        first_peak_left_idx = main_peak_idx - int(np.round(1.5 * num_samples_null))
        first_peak_right_idx = main_peak_idx + int(np.round(1.5 * num_samples_null))
        
        search_samples_left_stop = first_peak_left_idx - num_samples_search
        search_samples_right_stop = first_peak_right_idx + num_samples_search

        samples_left = matched_output[first_peak_left_idx : search_samples_left_stop : -1]
        samples_right = matched_output[first_peak_right_idx : search_samples_right_stop]
    else:
        raise Exception("The variable num_nulls_main cannot be greater than 2.")
    
    #Search for left null
    diffsign_left = np.sign(np.diff(samples_left))
    if np.any(diffsign_left == 1):
        null_left_idx = first_peak_left_idx - np.where(diffsign_left[:-1] + diffsign_left[1:] == 0)[0][0] - 1
    else:
        null_left_idx = first_peak_left_idx - num_samples_null
            
    #Search for right null
    diffsign_right = np.sign(np.diff(samples_right))
    if np.any(diffsign_right == -1):
        null_right_idx = first_peak_right_idx + np.where(diffsign_right[:-1] + diffsign_right[1:] == 0)[0][0] + 1
    else:
        null_right_idx = first_peak_right_idx + num_samples_null
    
    return null_left_idx, null_right_idx

def islr_pslr(data_in_linear, fs_bw_ratio=1.2, num_nulls_main=2, num_lobes=12, search_null=False):
    """Compute point target integrated sidelobe ratio (ISLR) and peak to sidelobe ratio (PSLR).
    
    Parameters:
    -----------
    fs_bw_ratio: float, optional, sampling frequency to bandwidth ratio
    search_null: if search_null is True, then apply algorithm to find mainlobe null locations
        for ISLR computation. Otherwise, specify null locations based on default Fs/B samples,
        i.e, mainlobe null is located at Fs/B samples from the peak of mainlobe
        PSLR Exception: mainlobe does not include first sidelobe, search is always
        conducted to find the locations of first null regardless of search_null parameter.
    num_nulls_main: int, optional maximum is 2. Mainlobe could include up to 2 nulls.
    num_lobes: float, optional total number of sidelobes for ISLR computation,
        if num_nulls_main=2,default is 12. If num_nulls_main=1, default is 11.
    data_in_linear: complex array, Linear Point target range or azimuth cut in complex numbers

    Returns:
    --------
    1. islr_dB: float, ISLR in dB
    2. pslr_dB: float, PSLR in dB
    """
    
    num_samples_null = int(np.round(fs_bw_ratio))
    data_in_pwr_linear = np.abs(data_in_linear)**2
    data_in_pwr_dB = 10*np.log10(data_in_pwr_linear)
    zmax_idx = np.argmax(data_in_pwr_linear)
    plsr_main_lobe = 1
    
    if search_null:
        null_main_left_idx, null_main_right_idx = find_null_to_null(data_in_pwr_dB, num_nulls_main, num_samples_null, zmax_idx)
        null_first_left_idx, null_first_right_idx = find_null_to_null(data_in_pwr_dB, plsr_main_lobe, num_samples_null, zmax_idx)
		
        num_samples_sidelobe = zmax_idx - null_first_left_idx
        num_samples_side_total = int(np.round(num_lobes * num_samples_sidelobe))
    else:
        num_samples_search = int(np.round(num_nulls_main * fs_bw_ratio))
        null_main_left_idx = zmax_idx - num_samples_search
        null_main_right_idx = zmax_idx + num_samples_search

        null_first_left_idx, null_first_right_idx = find_null_to_null(data_in_pwr_dB, plsr_main_lobe, num_samples_null, zmax_idx)
        num_samples_sidelobe = zmax_idx - null_first_left_idx
        num_samples_side_total = int(np.round(num_lobes * num_samples_sidelobe))
  
    sidelobe_left_idx = null_first_left_idx - num_samples_side_total
    sidelobe_right_idx = null_first_right_idx + num_samples_side_total

    #ISLR: Mainlobe could include 2nd null
    islr_mainlobe = data_in_pwr_linear[null_main_left_idx : null_main_right_idx + 1]
    
    islr_sidelobe_range = np.r_[sidelobe_left_idx : null_main_left_idx, null_main_right_idx + 1 : sidelobe_right_idx + 1]
    islr_sidelobe = data_in_pwr_linear[islr_sidelobe_range]   
    
    pwr_total = np.sum(data_in_pwr_linear)
    islr_main_pwr = np.sum(islr_mainlobe)
    islr_side_pwr = np.sum(islr_sidelobe)

    islr_dB = 10*np.log10(islr_side_pwr / islr_main_pwr)

    #PSLR
    pslr_sidelobe_range = np.r_[sidelobe_left_idx : null_first_left_idx, null_first_right_idx + 1 : sidelobe_right_idx + 1]
    pslr_main_lobe = data_in_pwr_linear[null_first_left_idx : null_first_right_idx]
    pslr_side_lobe = data_in_pwr_linear[pslr_sidelobe_range]
    
    pwr_main_max = np.amax(pslr_main_lobe)
    pwr_side_max = max(pslr_side_lobe)
    
    pslr_dB = 10*np.log10(pwr_side_max / pwr_main_max)
    
    return islr_dB, pslr_dB
	

def dB (x):
    return 20.0 * np.log10 (abs (x))


def plot_profile (t, x, title=None):
    import matplotlib.pyplot as plt

    peak = abs (x).max()
    fig = plt.figure()
    ax1 = fig.add_subplot (111)
    ax1.plot (t, dB(x) - dB(peak), '-k')
    ax1.set_ylim ((-40,0.3))
    ax1.set_ylabel ("Power (dB)")
    ax2 = ax1.twinx()
    phase_color = '0.75'
    ax2.plot (t, np.angle (x), color=phase_color)
    ax2.set_ylim ((-np.pi, np.pi))
    ax2.set_ylabel ("Phase (rad)")
    ax2.spines['right'].set_color(phase_color)
    ax2.tick_params (axis='y', colors=phase_color)
    ax2.yaxis.label.set_color (phase_color)
    ax1.set_xlim ((-15,15))
    ax1.spines['top'].set_visible (False)
    ax2.spines['top'].set_visible (False)
    if title:
        ax1.set_title (title)
    return fig


def analyze_point_target (slc, i, j, nov=32, plot=False, cuts=False,
                          chipsize=64, fs_bw_ratio=1.2, num_nulls_main=2, num_lobes=12, search_null=False):
    """Measure point-target attributes.

    Inputs:
        slc
            Single look complex image (2D array).

        i, j
            Row and column indeces where point-target is expected.

        nov
            Amount of oversampling.

        plot
            Generate interactive plots.

        cuts
            Include cuts through the peak in the output dictionary.

    Outputs:
        Dictionary of point target attributes.  If plot=true then return the
        dictionary and a list of figures.
    """

    chip_i0, chip_j0, chip = get_chip (slc, i, j, nchip=chipsize)

    chip, fx, fy = oversample (chip, nov=nov, return_slopes=True)

    k = np.argmax (abs (chip))
    ichip, jchip = np.unravel_index (k, chip.shape)
    chipmax = chip[ichip,jchip]

    imax = chip_i0 + ichip * 1.0/nov
    jmax = chip_j0 + jchip * 1.0/nov

    az_slice = chip[:,jchip]
    rg_slice = chip[ichip,:]

    dr = estimate_resolution (rg_slice, 1.0/nov)
    da = estimate_resolution (az_slice, 1.0/nov)

    # Find PSLR and ISLR of range and azimuth cuts
    fs_bw_ratio_ov = nov * fs_bw_ratio
    dr_islr_dB, dr_pslr_dB = islr_pslr(rg_slice, fs_bw_ratio=fs_bw_ratio_ov, num_nulls_main=num_nulls_main, num_lobes=num_lobes, search_null=search_null)
    da_islr_dB, da_pslr_dB = islr_pslr(az_slice, fs_bw_ratio=fs_bw_ratio_ov, num_nulls_main=num_nulls_main, num_lobes=num_lobes, search_null=search_null)
    
    d = {
        'magnitude': abs (chipmax),
        'phase': np.angle (chipmax),
        'range': {
            'index': jmax,
            'offset': jmax-j,
            'phase ramp': fx,
            'resolution': dr,
            'ISLR': dr_islr_dB,
            'PSLR': dr_pslr_dB,
        },
        'azimuth': {
            'index': imax,
            'offset': imax-i,
            'phase ramp': fy,
            'resolution': da,
            'ISLR': da_islr_dB,
            'PSLR': da_pslr_dB,
        },
    }

    idx = np.arange (chip.shape[0], dtype=float)
    ti = chip_i0 + idx/nov - i
    tj = chip_j0 + idx/nov - j
    if cuts:
        d['range']['magnitude cut'] = list (np.abs (rg_slice))
        d['range']['phase cut'] = list (np.angle (rg_slice))
        d['range']['cut'] = list (tj)
        d['azimuth']['magnitude cut'] = list (np.abs (az_slice))
        d['azimuth']['phase cut'] = list (np.angle (az_slice))
        d['azimuth']['cut'] = list (ti)
    if plot:
        figs = [plot_profile (tj, rg_slice, title='Range'),
                plot_profile (ti, az_slice, title='Azimuth')]
        return d, figs
    return d


def tofloatvals (x):
    """Map all values in a (possibly nested) dictionary to Python floats.

    Modifies the dictionary in-place and returns None.
    """
    for k in x:
        if type(x[k]) == dict:
            tofloatvals (x[k])
        elif type(x[k]) == list:
            x[k] = [float(xki) for xki in x[k]]
        else:
            x[k] = float (x[k])


def main (argv):
    from argparse import ArgumentParser
    import matplotlib.pyplot as plt
    import json

    parser = ArgumentParser (description=desc)
    parser.add_argument ('-1', action='store_true', dest='one_based',
                         help='Use one-based (Fortran) indexes.')
    parser.add_argument ('-i', action='store_true', help='Interactive plots.')
    parser.add_argument ('--cuts', action='store_true',
                         help='Add range/azimuth slices to output JSON.')
    parser.add_argument ('--chipsize', type=int, default=64)
    parser.add_argument ('filename')
    parser.add_argument ('n', type=int)
    parser.add_argument ('row', type=float)
    parser.add_argument ('column', type=float)
    parser.add_argument ('--fs-bw-ratio', type=float, default=1.2,
                         required=False, help='nisar oversampling ratio')
    parser.add_argument ('--mlobe-nulls', type=int, default=2,
                         required=False, help='number of nulls in mainlobe, default=2')
    parser.add_argument ('--num-lobes', type=float, default=12,
                         required=False, help='total number of lobes, including mainlobe, default=12')
    parser.add_argument('-s', '--search-null',
                         action='store_true', default='False', help='Search for mainlobe null or use default mainlobe sample spacing')
    args = parser.parse_args (argv[1:])

    n, i, j = [getattr (args,x) for x in ('n', 'row', 'column')]
    if args.one_based:
        i, j = i-1, j-1

    x = np.memmap (args.filename, dtype='complex64', mode='r')
    m = len (x) // n
    x = x.reshape ((m,n))

    info = analyze_point_target (x, i, j, plot=args.i, cuts=args.cuts,
                                 chipsize=args.chipsize, fs_bw_ratio=args.fs_bw_ratio,
                                 num_nulls_main=args.mlobe_nulls, num_lobes=args.num_lobes, search_null=args.search_null)
    if args.i:
        info = info[0]

    tofloatvals (info)
    print (json.dumps (info, indent=2))

    if args.i:
        plt.show()


if __name__ == '__main__':
    main (sys.argv)
