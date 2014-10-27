# Licensed under a 3-clause BSD style license - see LICENSE

__all__ = ["Star_temperature"]

from ..Utils.import_modules import *
from ..Utils import Spherical_harmonics
from .Star import Star


######################## class Star_temperature ########################
class Star_temperature(Star):
    """Star_temperature(Star)
    This class allows to determine the flux of a star
    in a binary system using an atmosphere grid. It is derived
    from the Star class.
    
    The noticeable difference is that this class contains the tools
    to play with the temperature distribution at the surface using
    spherical harmonics. This is an empirical way of reconstructing
    lightcurves that present peculiar behaviours such at asymmetries.
    """
    def __init__(self, *args, **kwargs):
        Star.__init__(self, *args, **kwargs)
        # We need the theta and phi in order to evaluate the spherical harmonics
        self.theta = np.arccos(self.cosz)
        self.phi = np.arctan2(self.cosy,self.cosx)

    def _Calc_teff(self, temp=None, tirr=None):
        """_Calc_teff(temp=None, tirr=None)
        Calculates the log of the effective temperature on the
        various surface elements of the stellar surface grid.
        
        For surface elements that are not exposed to the primary's
        irradiated flux, the base temperature profile is described
        using spherical harmonics. This temperature is then modified
        by a factor that takes into account the gravity darking
        through the 'tempgrav' exponent specified when calling
        Make_surface().
        
        For the exposed surface elements, an irradiation temperature
        is added the base temperature as:
            (Tbase**4 + coschi/r**2*Tirr**4)**0.24
        where 'coschi' is the angle between the normal to the surface
        element and the direction to the irradiation source and 'r'
        is the distance.
        
        Note: For an empirical approach, i.e. modelling the stellar
        temperature profile using spherical harmonics only, one can
        set tempgrav = 0., which disables the gravity darkening, and
        also set tirr = 0., which disables the irradiation.
        
        temp (None): Base temperature of the star.
            Temperature profile determined by spherical harmonic
            having the coefficients listed in 'temp'.
            Must have the form:
            [A_{00},A_{1-1},A_{10},A_{11}, ]
            If None, will use self.temp, otherwise, will use
            temp and set self.temp = temp.
        tirr (None): Irradiation temperature of the star.
            (lirr = eff * edot / (4*PI * a**2 * sigma))
            (tirr = (eff * edot / (4*PI * a**2 * sigma))**0.25)
            If None, will use self.tirr, otherwise, will use
            tirr and set self.tirr = tirr.
        
        >>> self._Calc_teff()
        """
        if temp is not None:
            self.temp = temp
        if tirr is not None:
            self.tirr = tirr
        # We calculate the base temperature profile using spherical harmonics
        teff = Spherical_harmonics.Composition(self.temp, self.phi, self.theta)
        # We calculate the gravity darkening correction to the temperatures across the surface and multiply them by the base temperature. We only do it if the gravity darkening is enabled (i.e. tempgrav != 0.).
        if self.tempgrav != 0:
            teff *= self._Gravdark()
        # We apply the irradiation to the surface visible to the irradiation source. Only if there are surface elements facing the irradiation source and if the irradiation is not null.
        if self.tirr != 0.:
            inds = self.coschi > 0
            if inds.any():
                teff[inds] = (teff[inds]**4+self.coschi[inds]*self.tirr**4/self.rx[inds]**2)**0.25
        self.logteff = np.log(teff)
        return

    def Spherical_coefficients(self, lmax, ndigit=None, verbose=True):
        """Spherical_coefficients(lmax, ndigit=None, verbose=True)
        Returns the spherical harmonic coefficients for the current
        temperature distribution.
        
        lmax: Maximum l number of coefficients.
        ndigit (None): if not None, will round off the results at
            ndigit (as per the np.round function).
        verbose (True): Prints the resulting coefficients.
        
        >>> alm = self.Spherical_coefficients(lmax)
        """
        alm = Spherical_harmonics.Decomposition(lmax, self.phi, self.theta, np.exp(self.logteff), ndigit=ndigit)
        if verbose:
            Spherical_harmonics.Pretty_print_alm(alm)
        return alm

######################## class Star_temperature ########################

