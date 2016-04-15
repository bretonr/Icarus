# Licensed under a 3-clause BSD style license - see LICENSE

from .import_modules import *


##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##
## Flux utilities
## Contain functions that pertain "flux-related" purposes
## such as flux to mag conversion, extinction,  etc.
##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##


def Asinh_to_flux(mag, mag_err=None, zeropoint=0., flux0=1., softening=1.):
    """ Asinh_to_flux(mag, mag_err=None, zeropoint=0., flux0=1., softening=1.)
    Converts the asinh magnitude to flux and its error, if provided.
    If the softening parameter is not provided, the value will be
    assumed to be 1., which might be far off.
    
    See Lupton, Gunn and Szalay 1999 (1999AJ....118.1406L).
    
    mag: magnitude
    mag_err (None): magnitude error
    zeropoint (0.): zero-point value, in magnitude
    flux0 (1.): zero-point value, in flux
    softening (1.): softening parameter. If nothing is provided, it will
        be assumed to be 1. Careful as it might be far off.
    
    Either zeropoint or flux0 can be provided. By default, the values have no
    effect. Make sure that both are not provided otherwise they might conflict.
    
    >>> flux,flux_err = Asinh_to_flux(10., 0.1, 0.)
    >>> flux = Asinh_to_flux(10., softening=1)
    """
    # Defining the Pogson's constant
    pogson = 2.5*np.log10(np.e)
    # Calculate the fluxes
    zeropoint = zeropoint + 2.5*np.log10(flux0)
    flux = 2*softening * np.sinh( ((zeropoint - 2.5*np.log10(softening)) - mag)/pogson )
    if mag_err is None:
        return flux
    else:
        # Calculate the flux errors
        flux_err = mag_err * 2*softening/pogson * np.sqrt( 1 + (flux/(2*softening))**2 )
        return flux, flux_err

def Distance_modulus_to_distance(dm, absorption=0.0):
    """
    Returns the distance in kpc for a distance modulus.

    dm (float): distance modulus.
    absorption (float): absorption to the source.
    """
    distance = 10.**(((dm-absorption)+5.)/5.) / 1000.
    return distance

def Distance_to_distance_modulus(distance, absorption=0.0):
    """
    Returns the distance modulus for a distance in kpc.

    distance (float): distance in kpc.
    absorption (float): absorption to the source.
    """
    dm = 5.0*np.log10(distance*1000.) - 5. + absorption
    return dm

def Extinction(w, Rv=3.1, cardelli=False):
    """Extinction(w, Rv=3.1)
    Returns the selective extinction for A_V=1 for a set of wavelengths w.
    
    w: wavelengths in microns.
    Rv (3.1): ratio of total to selective extinction (A_V/E(B-V)).
        standard value is Rv=3.1.
    cardelli (False): If true will use Cardelli et al. 1989.
        If false, will use O'Donnell 1994.
    
    
    Source:
    Cardelli, Clayton and Mathis, 1989, ApJ, 345, 245:
    http://adsabs.harvard.edu/abs/1989ApJ...345..245C
    
    O'Donnell, J. E., 1994, ApJ, 422, 158:
    http://adsabs.harvard.edu/abs/1994ApJ...422..158O
    Note: 1/w values between 1.1 micron^-1 < x < 3.3 microns^-1 (near-IR, optical)
    
    """
    x = np.atleast_1d(1/w)
    ext = np.zeros_like(x)
    #
    inds = x < 0.3
    ext[inds] = np.nan
    #
    inds = (x >= 0.3) * (x < 1.1)
    if inds.any():
        y = x[inds]
        p_a = 0.574 * y**1.61
        p_b = -0.527 * y**1.61
        ext[inds] = p_a + p_b/Rv
    #
    inds = (x >= 1.1) * (x < 3.3)
    if inds.any():
        y = x[inds]-1.82
        ##### Using Cardelli, Clayton and Mathis (1989)
        if cardelli:
            p_a = np.poly1d([0.32999, -0.77530, 0.01979, 0.72085, -0.02427, -0.50447, 0.17699, 1.])(y)
            p_b = np.poly1d([-2.09002, 5.30260, -0.62251, -5.38434, 1.07233, 2.28305, 1.41338, 0.])(y)
        ##### Using O'Donnell (1994)
        else:
            p_a = np.poly1d([-0.505, 1.647, -0.827, -1.718, 1.137, 0.701, -0.609, 0.104, 1.])(y)
            p_b = np.poly1d([3.347, -10.805, 5.491, 11.102, -7.985, -3.989, 2.908, 1.952, 0.])(y)
        ext[inds] = p_a + p_b/Rv
    #
    inds = (x >= 3.3) * (x < 5.9)
    if inds.any():
        y = x[inds]
        p_a = 1.752 - 0.316*y - 0.104/((y-4.67)**2 + 0.341)
        p_b = -3.090 + 1.825*y + 1.206/((y-4.62)**2 + 0.263)
        ext[inds] = p_a + p_b/Rv
    #
    inds = (x >= 5.9) * (x < 8.0)
    if inds.any():
        y = x[inds]
        f_a = -0.04473*(y-5.9)**2 - 0.009779*(y-5.9)**3
        f_b = 0.2130*(y-5.9)**2 + 0.1207*(y-5.9)**3
        p_a = 1.752 - 0.316*y - 0.104/((y-4.67)**2 + 0.341)
        p_b = -3.090 + 1.825*y + 1.206/((y-4.62)**2 + 0.263)
        ext[inds] = (p_a + f_a) + (p_b + f_b)/Rv
    #
    inds = (x >= 8.0) * (x <= 10.0)
    if inds.any():
        y = x[inds]
        p_a = -1.073 - 0.628*(y-8) + 0.137*(y-8)**2 - 0.070*(y-8)**3
        p_b = 13.670 + 4.257*(y-8) - 0.420*(y-8)**2 + 0.374*(y-8)**3
        ext[inds] = p_a + p_b/Rv
    #
    inds = x > 10.0
    ext[inds] = np.nan
    
    ##### Return scalar if possible
    if ext.shape == (1,):
        return ext[0]
    return ext

def Flux_to_asinh(flux, flux_err=None, zeropoint=0., flux0=1., softening=None):
    """ Flux_to_asinh(flux, flux_err=None, zeropoint=0., flux0=1., softening=None)
    Converts the flux to asinh magnitude and its error, if provided.
    If the softening parameter is not provided, the value will be
    determined from the flux_err, hence if none is present the function
    will crash.
    
    See Lupton, Gunn and Szalay 1999 (1999AJ....118.1406L).
    
    flux: flux
    flux_err (None): flux error
    zeropoint (0.): zero-point value, in magnitude, added to the magnitude
    flux0 (1.): zero-point value, in flux
    softening (None): softening parameter. If none is provided, it will
        be calculated as: softening = flux_err * sqrt(2.5*log10(e))
    
    Either zeropoint or flux0 can be provided. By default, the values have no
    effect. Make sure that both are not provided otherwise they might conflict.
    
    >>> mag,mag_err = Flux_to_asinh(10., 1., 0.)
    >>> mag = Flux_to_asinh(10., softening=1)
    """
    # Defining the Pogson's constant
    pogson = 2.5*np.log10(np.e)
    # Making sure that we can define the softening parameter
    if flux_err is None and softening is None:
        raise RuntimeError("Either flux_err or softening must be provided!")
    # Automatically infer the softening parameter if needed
    if softening is None:
        softening = flux_err * np.sqrt(pogson)
    # Calculate the magnitudes
    zeropoint = zeropoint + 2.5*np.log10(flux0)
    mag = (zeropoint - 2.5*np.log10(softening)) - pogson * np.arcsinh(flux/(2*softening))
    if flux_err is None:
        return mag
    else:
        # Calculate the magnitude errors
        mag_err = pogson / (2 * softening) * flux_err / np.sqrt( 1 + (flux/(2*softening))**2 )
        return mag, mag_err

def Flux_to_mag(flux, flux_err=None, zeropoint=0., flux0=1.):
    """ Flux_to_mag(flux, flux_err=None, zeropoint=0., flux0=1.)
    Converts the flux to magnitude and its error, if provided.
    
    flux: flux in erg/s/cm^2/Hz (note 1 Jy = 1e-23 erg/s/cm^2/Hz)
    flux_err (None): flux error
    zeropoint (0.): zero-point value, in magnitude, added to the magnitude
    flux0 (1.): zero-point value, in flux
        in erg/s/cm^2/Hz
    
    Either zeropoint or flux0 can be provided. By default, the values have no
    effect. Make sure that both are not provided otherwise they might conflict.
    
    >>> mag,mag_err = Flux_to_mag(10., 1., 0.)
    >>> mag = Flux_to_mag(10.)
    """
    mag = -2.5 * np.log10(flux/flux0) + zeropoint
    if flux_err is not None:
        mag_err = 2.5 * np.log10(np.e) * flux_err / flux
        return mag, mag_err
    else:
        return mag

def Limb_darkening(lam, mu):
    """
    Returns the limb darkening.
    
    lam: wavelength in micrometer.
        nlam = lam.shape
    mu: cos(theta) direction of emission angle.
        nmu, 1 = mu.shape
    
    Note: Only valid for 0.42257 < lam < 1.100 micrometer.
    From Neckel 2005.
    """
    def L_422_1100(lam,mu):
        a_00 = 0.75267
        a_01 = -0.265577
        a_10 = 0.93874
        a_11 = 0.265577
        a_15 = -0.004095
        a_20 = -1.89287
        a_25 = 0.012582
        a_30 = 2.42234
        a_35 = -0.017117
        a_40 = -1.71150
        a_45 = 0.011977
        a_50 = 0.49062
        a_55 = -0.003347
        lam5 = lam**5
        a_0 = a_00 + a_01/lam
        a_1 = a_10 + a_11/lam + a_15/lam5
        a_2 = a_20 + a_25/lam5
        a_3 = a_30 + a_35/lam5
        a_4 = a_40 + a_45/lam5
        a_5 = a_50 + a_55/lam5
        return a_0 + (a_1 + (a_2 + (a_3 + (a_4 + a_5*mu )*mu )*mu )*mu )*mu
    def L_385_422(lam,mu):
        a_00 = 0.09900
        a_01 = 0.010643
        a_10 = 1.96884
        a_11 = -0.010643
        a_15 = -0.009166
        a_20 = -2.80511
        a_25 = 0.024873
        a_30 = 3.32780
        a_35 = -0.029317
        a_40 = -2.17878
        a_45 = 0.018273
        a_50 = 0.58825
        a_55 = -0.004663
        lam5 = lam**5
        a_0 = a_00 + a_01/lam
        a_1 = a_10 + a_11/lam + a_15/lam5
        a_2 = a_20 + a_25/lam5
        a_3 = a_30 + a_35/lam5
        a_4 = a_40 + a_45/lam5
        a_5 = a_50 + a_55/lam5
        return a_0 + (a_1 + (a_2 + (a_3 + (a_4 + a_5*mu )*mu )*mu )*mu )*mu
    def L_300_372(lam,mu):
        a_00 = 0.35601
        a_01 = -0.085217
        a_10 = 1.11529
        a_11 = 0.085217
        a_15 = -0.001871
        a_20 = -0.67237
        a_25 = 0.003589
        a_30 = 0.18696
        a_35 = -0.002415
        a_40 = 0.00038
        a_45 = 0.000897
        a_50 = 0.01373
        a_55 = -0.000200
        lam5 = lam**5
        a_0 = a_00 + a_01/lam
        a_1 = a_10 + a_11/lam + a_15/lam5
        a_2 = a_20 + a_25/lam5
        a_3 = a_30 + a_35/lam5
        a_4 = a_40 + a_45/lam5
        a_5 = a_50 + a_55/lam5
        return a_0 + (a_1 + (a_2 + (a_3 + (a_4 + a_5*mu )*mu )*mu )*mu )*mu
    limb = np.empty((mu.shape[0],lam.shape[0]))
    inds = lam<0.37298
    if inds.any():
        limb[:,inds] = L_300_372(lam[inds],mu)
    inds = (lam<0.42257)*(lam>0.37298)
    if inds.any():
        limb[:,inds] = L_385_422(lam[inds],mu)
    inds = lam>0.42257
    if inds.any():
        limb[:,inds] = L_422_1100(lam[inds],mu)
    return limb

def Mag_to_flux(mag, mag_err=None, zeropoint=0., flux0=1.):
    """ Mag_to_flux(mag, mag_err=None, zeropoint=0., flux0=1.)
    Converts the flux to magnitude and its error, if provided.
    Fluxes are in erg/s/cm^2/Hz (note 1 Jy = 1e-23 erg/s/cm^2/Hz)
    
    mag: magnitude
    mag_err (None): magnitude error
    zeropoint (0.): zero-point value, in magnitude
    flux0 (1.): zero-point value, in flux
        in erg/s/cm^2/Hz

    Either zeropoint or flux0 can be provided. By default, the values have no
    effect. Make sure that both are not provided otherwise they might conflict.
    
    >>> flux,flux_err = Mag_to_flux(10., 1., 0.)
    >>> flux = Mag_to_flux(10.)
    """
    flux = 10**(-(mag-zeropoint)/2.5) * flux0
    if mag_err is not None:
        flux_err = mag_err*flux/(2.5*np.log10(np.e))
        return flux, flux_err
    else:
        return flux


