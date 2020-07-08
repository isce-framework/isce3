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
                          chipsize=64):
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

    d = {
        'magnitude': abs (chipmax),
        'phase': np.angle (chipmax),
        'range': {
            'index': jmax,
            'offset': jmax-j,
            'phase ramp': fx,
            'resolution': dr,
        },
        'azimuth': {
            'index': imax,
            'offset': imax-i,
            'phase ramp': fy,
            'resolution': da,
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
    args = parser.parse_args (argv[1:])

    n, i, j = [getattr (args,x) for x in ('n', 'row', 'column')]
    if args.one_based:
        i, j = i-1, j-1

    x = np.memmap (args.filename, dtype='complex64', mode='r')
    m = len (x) // n
    x = x.reshape ((m,n))

    info = analyze_point_target (x, i, j, plot=args.i, cuts=args.cuts,
                                 chipsize=args.chipsize)
    if args.i:
        info = info[0]

    tofloatvals (info)
    print (json.dumps (info, indent=2))

    if args.i:
        plt.show()


if __name__ == '__main__':
    main (sys.argv)
