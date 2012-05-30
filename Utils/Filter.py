# Licensed under a 3-clause BSD style license - see LICENSE

from .import_modules import *


def Band_integration(band_func, w, f, nu=True):
    """ Band_integration(band_func, w, f, nu=True)
    
    band_func: function that interpolates the filter response at a given
        set of wavelengths/frequencies.
    w: wavelengths or frequencies of the source to be integrated in A or Hz.
    f: flux density in erg/s/cm^2/A or erg/s/cm^2/Hz.
    nu (True): Whether the integration should be performed in the frequency
        or wavelength domain.
    
    See The Alhambra Photometric System (doi:10.1088/0004-6256/139/3/1242) for more details.
    See also The Mauna Kea Observatories Near-Infrared Filter Set. III. Isophotal Wavelengths and Absolute Calibration (doi:10.1086/429382).
    """
    # Check if we work in the AB system (F_nu) or ST system (F_lambda)
    if nu:
        # Multiply the atmosphere data by the pass band.
        f_band = band_func(w)
        f_int = scipy.integrate.trapz(f_band*f/w, w) / scipy.integrate.trapz(f_band/w,w)
    else:
        # Multiply the atmosphere data by the pass band.
        f_band = band_func(w)
        f_int = scipy.integrate.trapz(f_band*f*w, w) / scipy.integrate.trapz(f_band*w,w)
    return f_int

def Bandint_BTSettl7(atmo_flns, band_fln, output_fln, nu=True):
    """ Bandint_BTSettl7(atmo_flns, band_fln, output_fln, nu=True)
    Integrate BTSettl7 atmosphere models through a given filter.
    
    atmo_flns: atmosphere model filenames.
    band_fln: filter filename.
    output_fln: output filename.
    nu (True): Whether the integration should be performed in the frequency
        or wavelength domain.
    """
    # Load the filter interpolator
    band_func = Load_filter(band_fln, nu=nu)
    
    # Iterating through the atmosphere data to integrate through the filter
    atmo_flns.sort()
    output = open(output_fln, 'w')
    for atmo_fln in atmo_flns:
        print( 'Integrating %s' %atmo_fln )
        ind = atmo_fln.find('lte')
        temp = float(atmo_fln[ind+3:ind+6])*100
        logg = float(atmo_fln[ind+7:ind+10])
        if atmo_fln[ind+6] == '+':
            logg *= -1
        w, f = Load_BTSettl7(atmo_fln, nu=nu)
        flux = Band_integration(band_func, w, f, nu=nu)
        output.write( '%7.1f %4.2f %e\n' %(temp,logg,flux) )
    output.close()
    return

def Load_BTSettl7(atmo_fln, nu=True):
    """ Load_BTSettl7(atmo_fln, nu=True)
    
    atmo_fln: BT-Settl.7 filename.
    nu (True): if True: returns w, f in Hz, erg/s/cm^2/Hz
        if False: return w, f in A, erg/s/cm^2/A
    """
    # Load the atmosphere data, first column is wavelength in A, second column if log10(flux)-8, in ergs/s/cm**2/A
    atmo = numpy.loadtxt(atmo_fln, unpack=True, usecols=(0,1))
    # We sort the atmosphere data by wavelength because they are not always in the right order.
    w = atmo[0,atmo[0].argsort()] # in angstrom
    f = atmo[1,atmo[0].argsort()]
    f = 10**(f-8) # in erg/s/cm^2/A
    if nu:
        # We convert to frequency and F_nu
        f = f * w**2 / cts.c / 1e10 # F_nu = F_lambda * lambda^2 / c / 10^10 => [erg/s/cm^2/A] [A^2] [m/s] = [erg/s/cm^2/Hz]
        w = cts.c / (w * 1e-10) # [w] = Hz
        f = f[::-1]
        w = w[::-1]
    return w, f

def Load_filter(band_fln, nu=True):
    """ Load_filter(band_fln, nu=True)
    Returns a function that interpolates the filter response at a given
    wavelength/frequency.
    
    band_fln: filter filename.
        The format should be two columns (wavelengths in A, response)
        The wavelengths must be in ascending order.
    nu (True): Whether the integration should be performed in the frequency
        or wavelength domain.
    """
    # Load the pass band data, first column is wavelength in A, second column is transmission
    w_filter, t_filter  = numpy.loadtxt(band_fln, unpack=True)[:2]
    # The Bessell filter data are in nm, so need to multiply by 10
    if band_fln.find('bessell') != -1:
        w_filter *= 10
    # Check if we work in the AB system (F_nu) or ST system (F_lambda)
    if nu:
        t_filter = t_filter[::-1]
        w_filter = cts.c / (w_filter[::-1] * 1e-10) # [w_filter] = Hz
    # Define an interpolation function, such that the atmosphere data can be directly multiplied by the pass band.
    band_func = scipy.interpolate.interp1d(w_filter, t_filter, kind='cubic', bounds_error=False, fill_value=0.)
    return band_func

def W_effective(band_func, w, nu=True):
    """ W_effective(band_func, w, nu=True)
    
    band_func: function that interpolates the filter response at a given
        set of wavelengths/frequencies.
    w: wavelengths or frequencies of the source to be integrated in A or Hz.
    nu (True): Whether the integration should be performed in the frequency
        or wavelength domain.
    
    See The Alhambra Photometric System (doi:10.1088/0004-6256/139/3/1242) for more details.
    See also The Mauna Kea Observatories Near-Infrared Filter Set. III. Isophotal Wavelengths and Absolute Calibration (doi:10.1086/429382).
    """
    # Check if we work in the AB system (F_nu) or ST system (F_lambda)
    if nu:
        # Multiply the atmosphere data by the pass band.
        f_band = band_func(w)
        f_int = scipy.integrate.trapz(f_band, w) / scipy.integrate.trapz(f_band/w,w)
    else:
        # Multiply the atmosphere data by the pass band.
        f_band = band_func(w)
        f_int = scipy.integrate.trapz(f_band*w, w) / scipy.integrate.trapz(f_band,w)
    return f_int


