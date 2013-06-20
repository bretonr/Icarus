# Licensed under a 3-clause BSD style license - see LICENSE

from .import_modules import *
from . import Utils
try:
    import h5py
except:
    print( "Failed at importing h5py. Some data reading functionalities might not work properly." )


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

def Bandint_BTSettl_AGSS2009(atmo_flns, band_fln, output_fln, nu=True):
    """ Bandint_BTSettl_AGSS2009(atmo_flns, band_fln, output_fln, nu=True)
    Integrate BTSettl AGSS2009 atmosphere models (HDF format) through a
    given filter.
    
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
        w, f = Load_BTSettl_AGSS2009(atmo_fln, nu=nu)
        flux = Band_integration(band_func, w, f, nu=nu)
        output.write( '%7.1f %4.2f %e\n' %(temp,logg,flux) )
    output.close()
    return

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

def Doppler_boosting_BTSettl(atmo_flns, band_fln, maxv=500e3, deltav=1e3, resample=None, smooth=None, wrange=None, verbose=False):
    """Doppler_boosting_BTSettl(atmo_flns, band_fln, maxv=500e3, deltav=1e3, resample=None, smooth=None, wrange=None, verbose=False)
    This function calculates the Doppler boosting factor of BT-Settl spectra
    given a certain bandpass.

    atmo_flns (list): List of BT-Settl filenames. Can be h5, or .7 format.
    band_fln (str): Atmosphere filename.
    maxv (float): maximum value for the Doppler shift to be sampled in m/s
    deltav (float): Doppler shift sampling in m/s
    resample (float): If provided, will resample the spectrum at the value (in
        Armstroms).
    smooth (float): If provided, will convolve the spectrum with a gaussian kernel
        of sigma=smooth.
    wrange (list): One can provide the minimum and maximum value of wavelength
        to keep.
    verbose (bool): If true, will display the a plot of the fit for the Doppler boosting.
    """
    # Load the filter interpolator
    band_func = Load_filter(band_fln, nu=False)
    
    # Iterating through the atmosphere data to calculate the Doppler boosting
    boost = []
    atmo_flns.sort()
    for atmo_fln in atmo_flns:
        print( 'Doppler boosting for %s' %atmo_fln )
        if atmo_fln.find('.fits') != -1:
            import pyfits
            hdulist = pyfits.open(atmo_fln)
            hdu1 = hdulist[1]
            tbdata = hdu1.data
            f = tbdata.field(1)
            w = tbdata.field(0)
        else:
            w, f = Load_BTSettl_AGSS2009(atmo_fln, nu=False, wrange=wrange, resample=resample)
        if smooth is not None:
            f = scipy.ndimage.gaussian_filter1d(f, smooth)
        band = band_func(w)
        boost.append( Doppler_boosting_factor(f, band, w, verbose=verbose) )
        if verbose:
            nextplotpage()
    return numpy.array(boost)

def Doppler_boosting_factor(spectrum, bandpass, wavelengths, maxv=500e3, deltav=1e3, verbose=False):
    """Doppler_boosting_factor(spectrum, bandpass, wavelengths, maxv=500e3, deltav=1e3, verbose=False)
    This function calculates the Doppler boosting factor of a spectrum
    given a certain bandpass. Both spectrum and banpass must be sampled
    at the provided wavelengths values.

    spectrum (array): spectrum
    bandpass (array): bandpass of the filter
    wavelengths (array): wavelengths of the spectrum and bandpass
    maxv (float): maximum value for the Doppler shift to be sampled in m/s
    deltav (float): Doppler shift sampling in m/s
    verbose (bool): If true, will display the a plot of the fit for the Doppler boosting.
    """
    vels = numpy.arange(-maxv, maxv+deltav, deltav)
    wav0 = wavelengths[0]
    deltawav0 = wavelengths[1] - wavelengths[0]
    intflux = []
    for v in vels:
        spectrum_shifted = Utils.Shift_spectrum(spectrum, wavelengths, v, wav0, deltawav0)
        #intflux.append( numpy.sum(spectrum_shifted*bandpass*wavelengths*numpy.sqrt( (1-v/299792458.0)/(1+v/299792458.0) )**5) )
        intflux.append( numpy.sum(spectrum_shifted*bandpass*wavelengths / (1-v/cts.c)**5) )
    if verbose:
        plotxy(spectrum/spectrum.max(), wavelengths, rangey=[0,1.05])
        plotxy(bandpass/bandpass.max(), wavelengths, color=2)
        nextplotpage()
    intflux = numpy.array(intflux)
    intflux /= intflux[intflux.size/2]
    tmp = Utils.Fit_linear(intflux, x=vels/cts.c, b=1., output=verbose, inline=True)
    boost = tmp[1]
    return boost

def Load_BTSettl_AGSS2009(atmo_fln, nu=True, wrange=None, resample=None, debug=False):
    """ Load_BTSettl_AGSS2009(atmo_fln, nu=True, wrange=None, resample=None, debug=False)
    
    atmo_fln: BT-Settl.h5 filename.
    nu (True): if True: returns w, f in Hz, erg/s/cm^2/Hz
        if False: return w, f in A, erg/s/cm^2/A
    wrange (list): One can provide the minimum and maximum value of wavelength
        to keep.
    smooth (float): If provided, will convolve the spectrum with a gaussian kernel
        of sigma=smooth.
    """
    ## ascii format
    if atmo_fln.find('BT-Settl.7') != -1:
        ## Load the atmosphere data, first column is wavelength in A, second column if log10(flux), in ergs/s/cm**2/cm
        atmo = numpy.loadtxt(atmo_fln, unpack=True, usecols=(0,1), converters={0: lambda s: float(s.replace('D','E')), 1: lambda s: float(s.replace('D','E'))})
        f = atmo[1] # in log10(erg/s/cm^2/cm)
        f = 10**(f-8) # in erg/s/cm^2/A
        w = atmo[0] # in A
    ## h5 format
    elif atmo_fln[-11:] == 'BT-Settl.h5':
        atmo = h5py.File(atmo_fln, 'r')
        f = atmo['Spectrum']['flux'].value # in erg/s/cm^2/cm
        f = f*1e-8 # in erg/s/cm^2/A
        ## If the wavelength data exist we read them
        try:
            w = atmo['Spectrum']['wl'].value # in angstrom
        ## If it doesn't exist then we reconstruct it
        except:
            try:
                ber = atmo['Spectrum']['cmtber'].value
                dis = atmo['Spectrum']['cmtdis'].value
                ind = dis.tolist().index(0)
                npt = numpy.round((ber[1:ind+1]-ber[:ind]) / dis[:ind])
                w = [ numpy.linspace(ber[i], ber[i+1], npt[i], endpoint=False) for i in range(ind) ] + [ ber[ind:ind+1] ]
                w = numpy.concatenate(w)
            except:
                print( "Error constructing the wavelength grid from cmtber info for file {}".format(atmo_fln) )
                w = numpy.arange(f.size)
        f = numpy.asarray(f, dtype=float)
        w = numpy.asarray(w, dtype=float)
    else:
        print( "File format not recognized!" )
        return [], []
    ## We sort the atmosphere data by wavelength because they are not always in the right order.
    if wrange is not None:
        ind = (w > wrange[0]) * (w < wrange[1])
        w = w[ind]
        f = f[ind]
    #ind = w.argsort()
    #w = w[ind]
    #f = f[ind]
    ## Sometimes there are duplicated spectral bins. We weed them out and this takes care of sorting the bins in ascending order too.
    w, ind = numpy.unique(w, return_index=True)
    f = f[ind]
    ## We may want to resample the spectrum at a given resolution.
    if resample is not None:
        w_new = numpy.arange(w[0], w[-1], resample)
        weight, pos = Utils.Getaxispos_vector(w, w_new)
        f_new = f[pos]*(1-weight) + f[pos+1]*weight
        f = f_new
        w = w_new
    ## We convert to frequency and F_nu if requested
    if nu:
        f = f * w**2 / cts.c / 1e10 # F_nu = F_lambda * lambda^2 / c / 10^10 => [erg/s/cm^2/A] [A^2] [m/s] = [erg/s/cm^2/Hz]
        w = cts.c / (w * 1e-10) # [w] = Hz
        f = f[::-1]
        w = w[::-1]
    return w, f

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


