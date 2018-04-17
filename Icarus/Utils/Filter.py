# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function, division

from .import_modules import *
from . import Grid, Misc, Series


##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##
## Contain functions to perform tasks related to passband
## filters, such as flux integration.
##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##


def Band_integration(band_func, w, f, input_nu=False, AB=True, mask=None, method='simps'):
    """
    Integrate a spectrum over a filter response curve.

    band_func: function that interpolates the filter response at a given
        set of wavelengths/frequencies. Takes one parameter, which is the
        same units as w.
    w: wavelengths or frequencies of the source to be integrated in wavelength
        or frequency space.
        wavelengths must be in angstrom.
        wavelengths must be in Hz.
    f: flux density in erg/s/cm^2/A or erg/s/cm^2/Hz.
    input_nu: Whether the input band_func, w and f are in the frequency or
        wavelength domain.
    AB: Whether the integration should be performed in the STMAG system
        or the ABMAG system.
        (see equation 5,6 from Linnell, DeStefano & Hubeny, ApJ, 146, 68)
    mask: if a mask is provided, values that are non-zero will be masked from
        the integration.
    method: the integration method. Can be 'simps' for Simpson's rule or
        'trapz' for simple trapezoid. Defaults to 'simps'.

    See The Alhambra Photometric System (doi:10.1088/0004-6256/139/3/1242) for more details.
    See also The Mauna Kea Observatories Near-Infrared Filter Set. III. Isophotal Wavelengths and Absolute Calibration (doi:10.1086/429382).
    """
    ## Evaluate the band transmission at the given frequency/wavelength
    f_band = band_func(w)
    ## Check if we work in the AB system (F_nu)
    if AB:
        ## The following equation is from Bessell & Murphy 2012 (eq. A12a)
        ## <f_nu> from f_nu, S_nu and nu
        if input_nu:
            val_nominator = f*f_band/w
            val_denominator = f_band/w
            #f_int = scipy.integrate.simps(f*f_band/w, w) / scipy.integrate.simps(f_band/w, w)
        ## The following equation is from Bessell & Murphy 2012 (eq. A12b)
        ## <f_nu> from f_lambda, S_lambda and lambda
        ## Note that in order to balance, we must use the speed of light in A/s
        else:
            val_nominator = f*f_band*w
            val_denominator = f_band*(cts.c*1e10)/w
            #f_int = scipy.integrate.simps(f*f_band*w, w) / scipy.integrate.simps(f_band*(cts.c*1e10)/w, w)
    ## If not we work in the ST system (F_lambda)
    else:
        ## The following has not been implemented
        if input_nu:
        ## The following equation is inferred from Bessell & Murphy 2012 (eq. A11)
        ## <f_lambda> from f_nu, S_nu and nu
            val_nominator = f*f_band/w
            val_denominator = f_band*cts.c/w**3
            #f_int = scipy.integrate.simps(f*f_band/w, w) / scipy.integrate.simps(f_band*cts.c/w**3, w)
        ## The following equation is from Bessell & Murphy 2012 (eq. A11)
        ## <f_lambda> from f_lambda, S_lambda and lambda
        else:
            val_nominator = f*f_band*w
            val_denominator = f_band*w
            #f_int = scipy.integrate.simps(f*f_band*w, w) / scipy.integrate.simps(f_band*w, w)
        ## For the ST system (F_lambda), we convert back from m to A
        #f_int = f_int * 1e-10
    ## Determine if a mask is necessary
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        val_nominator[mask] = 0.
        val_denominator[mask] = 0.
    ## Set the integration method
    if method is 'trapz':
        dw = np.gradient(w)
        f_int = (val_nominator*dw).sum() / (val_denominator*dw).sum()
    else:
        f_int = scipy.integrate.simps(val_nominator, w) / scipy.integrate.simps(val_denominator, w)
    return f_int

def Doppler_boosting_factor(band_func, w, f, velocities, input_nu=False, AB=True):
    """
    This function calculates the Doppler boosting factor of a spectrum
    given a certain bandpass at the provided velocities.

    band_func: function that interpolates the filter response at a given
        set of wavelengths/frequencies. Takes one parameter, which is the
        same units as w.
    w: wavelengths or frequencies of the source to be integrated in wavelength
        or frequency space.
        wavelengths must be in angstrom.
        wavelengths must be in Hz.
    f: flux density in erg/s/cm^2/A or erg/s/cm^2/Hz.
    velocities: velocities at which the Doppler boosting should be calculated.
        Positive -> moving away from observer.
        Negative -> moving towards observer.
    input_nu: Whether the input band_func, w and f are in the frequency or
        wavelength domain.
    AB: Whether the integration should be performed in the STMAG system
        or the ABMAG system.
        (see equation 5,6 from Linnell, DeStefano & Hubeny, ApJ, 146, 68)

    See The Alhambra Photometric System (doi:10.1088/0004-6256/139/3/1242) for more details.
    See also The Mauna Kea Observatories Near-Infrared Filter Set. III. Isophotal Wavelengths and Absolute Calibration (doi:10.1086/429382).
    """
    velocities = np.atleast_1d(velocities)
    f0 = Band_integration(band_func, w, f, input_nu=input_nu, AB=AB)
    boost = np.empty((velocities.size, f0.size), dtype=float)
    for i, vel in enumerate(velocities):
        if input_nu:
            w_shifted = w * np.sqrt( (1.-vel/cts.c)/(1.+vel/cts.c) )
        else:
            w_shifted = w * np.sqrt( (1.+vel/cts.c)/(1.-vel/cts.c) )
        if vel == 0.:
            boost[i] = 1.
        else:
            boost[i] = Band_integration(band_func, w_shifted, f, input_nu=input_nu, AB=AB) / (1+vel/cts.c)**5 / f0
    return boost

def Load_filter(band_fln, conv=1., kind='quadratic', provide_bounds=False):
    """
    Returns a function that interpolates the filter response at a given
    wavelength/frequency.

    band_fln: filter filename.
        The format should be two columns
            wavelengths in A, response
            or
            frequency in Hz, response
    conv: the conversion factor to multiply the first column to get A or Hz.
    kind: the type of interpolation to use ('linear', 'quadratic', 'cubic')
    provide_bounds: if true, will return w.min() and w.max() as a tuple, in
        addition to the function.

    Examples
    --------
        band_func = Load_filter(fln)
        band_func, (wlow,whigh) = Load_filter(fln, provide_bounds=True)
    """
    ## Load the pass band data, first column is wavelength in A, second column is transmission
    w, t  = np.loadtxt(band_fln, unpack=True)[:2]
    ## We reorder the filter data to make sure that it is an increasing function of the wavelength
    inds = w.argsort()
    w = w[inds]
    t = t[inds]
    ## We multiply the wavelength by the conversion factor to ensure that it is in A
    band_func = scipy.interpolate.interp1d(w*conv, t, kind=kind, bounds_error=False, fill_value=0.)
    if provide_bounds:
        return band_func, (w.min(),w.max())
    return band_func

def Pivot_wavelength(band_func, w):
    """
    band_func: function that interpolates the filter response at a given
        set of wavelengths.
    w: wavelengths in A.

    See The Alhambra Photometric System (doi:10.1088/0004-6256/139/3/1242) for more details.
    See also The Mauna Kea Observatories Near-Infrared Filter Set. III. Isophotal Wavelengths and Absolute Calibration (doi:10.1086/429382).
    """
    ## The following equation is from Bessell & Murphy 2012 (eq. A15)
    f_band = band_func(w)
    f_pivot = scipy.integrate.simps(f_band*w, w) / scipy.integrate.simps(f_band/w, w)
    f_pivot = np.sqrt(f_pivot)
    return f_pivot

def Resample_grid(w, f, resample=None, wrange=None, axis=-1):
    """
    Takes a (spectral) grid f and the associated coordinates (e.g. wavelengths,
    frequencies) w and resample it using a simple linear interpolation. The
    grid can potentially be trimmed off as well.

    Parameters
    ----------
    w : ndarray
        Coordinates (e.g. wavelengths/frequencies).
    f : ndarray
        Grid (e.g. spectral fluxes). Can be N-dimensional.
    resample : None, float, ndarray
        New sampling, in the same units as the w parameter.
        If None is given, no resampling is performed (thus only trimming).
        If a float is given, the array will be resampled at a constant interval
        between the minimum and maximum coordinate value.
        If an vector array is given, the grid will be resampled at the given
        values.
    wrange : list
        Minimum and maximum value to trim the coordinates at.
        Note that the range is inclusive.
        If None is given, the current boundaries are preserved.
    axis : int
        In the case of an N-dimensional grid, index represents the axis along
        which to perform the interpolation.

    Returns
    -------
    w_new : ndarray
        Resampled coordinates (e.g. wavelengths, frequencies)
    f_new : ndarray
        Resampled grid (e.g. spectral fluxes)
    """
    ## Assert that the original coordinate matches the axis of the grid
    if w.size != f.shape[axis]:
        raise Exception("The length of w must match the dimension 'axis' of the grid f.")

    ## We may want to resample the spectrum at a given resolution.
    if wrange is not None:
        inds = (w>=wrange[0])*(w<=wrange[1])
        w = w[inds]
        if f.ndim == 1:
            f = f[inds]
        else:
            fslice = [slice(l) for l in f.shape]
            fslice[axis] = inds
            f = f[fslice]
    if resample is not None:
        if isinstance(resample, float):
            ## Because np.arange does not include the last point, we need to add
            ## the resampling size to catch the last element in case it falls on.
            w_new = np.arange(w[0], w[-1]+resample, resample)
        else:
            w_new = resample
        weight, pos = Series.Getaxispos_vector(w, w_new)

        if f.ndim == 1:
            f = f[pos]*(1-weight) + f[pos+1]*weight
            w = w_new
            return w_new, f
        else:
            if axis < 0:
                axis = f.ndim + axis
            if axis == f.ndim-1:
                f = f[...,pos]*(1-weight) + f[...,pos+1]*weight
            else:
                f = (f.swapaxes(axis,-1)[...,pos]*(1-weight) + f.swapaxes(axis,-1)[...,pos+1]*weight).swapaxes(axis,-1)
            return w_new, f
    return w, f

