# Licensed under a 3-clause BSD style license - see LICENSE

__all__ = ["Photometry_temperature"]

from ..Utils.import_modules import *
from .. import Core
from .Photometry import Photometry


######################## class Photometry_temperature ########################
class Photometry_temperature(Photometry):
    """Photometry_temperature
    This class allows to fit the flux from the primary star
    of a binary system, assuming it is heated by the secondary
    (which in most cases will be a pulsar).
    
    It is meant to deal with photometry data. Many sets of photometry
    data (i.e. different filters) are read. For each data set, one can
    calculate the predicted flux of the model at every data point (i.e.
    for a given orbital phase).
    """
    def __init__(self, atmo_fln, data_fln, ndiv, porb, x2sini, edot=1., read=True):
        """__init__(atmo_fln, data_fln, ndiv, porb, x2sini, edot=1., read=True)
        This class allows to fit the flux from the primary star
        of a binary system, assuming it is heated by the secondary
        (which in most cases will be a pulsar).
        
        It is meant to deal with photometry data. Many sets of photometry
        data (i.e. different filters) are read. For each data set, one can
        calculate the predicted flux of the model at every data point (i.e.
        for a given orbital phase).
        
        atmo_fln (str): A file containing the grid model information for each
            data set. The format of each line of the file is as follows:
                Col 0: band name
                Col 1: band filename
        data_fln (str): A file containing the information for each data set.
            Three formats are currently supported.
            9-column (preferred):
                Col 0: band name
                Col 1: column id for orbital phase. Orbital phases must be 0-1.
                    Phase 0 is defined as the primary star (the one modelled),
                    located at inferior conjunction.
                Col 2: column id for flux/magnitude
                Col 3: column id for flux/magnitude error
                Col 4: shift to phase zero. Sometimes people use other
                    definition for orbital phases, so this allows to correct for
                    it.
                Col 5: band calibration error
                Col 6: softening parameter for asinh magnitude conversion. If
                    the value is 0., then standard magnitudes are used.
                Col 7: flux or mag flag. Currently, all the data must be in the
                    same format.
                    'm' means magnitude system
                    'f' means flux system
                Col 8: filename
            8-column (support for asinh magnitudes, no fluxes input):
                Col 0: band name
                Col 1: column id for orbital phase. Orbital phases must be 0-1.
                    Phase 0 is defined as the primary star (the one modelled),
                    located at inferior conjunction.
                Col 2: column id for magnitude
                Col 3: column id for magnitude error
                Col 4: shift to phase zero. Sometimes people use other
                    definition for orbital phases, so this allows to correct for
                    it.
                Col 5: band calibration error
                Col 6: softening parameter for asinh magnitude conversion. If
                    the value is 0., then standard magnitudes are used.
                Col 7: filename
            7-column (only support standard magnitude input):
                Col 0: band name
                Col 1: column id for orbital phase. Orbital phases must be 0-1.
                    Phase 0 is defined as the primary star (the one modelled),
                    located at inferior conjunction.
                Col 2: column id for magnitude
                Col 3: column id for magnitude error
                Col 4: shift to phase zero. Sometimes people use other
                    definition for orbital phases, so this allows to correct for
                    it.
                Col 5: band calibration error
                Col 6: filename
        ndiv (int): The number of surface element subdivisions. Defines how
            coarse/fine the surface grid is.
        porb (float): Orbital period of the system in seconds.
        x2sini (float): Projected semi-major axis of the secondary (pulsar)
            in light-second.
        edot (float): Irradiated energy from the secondary, aka pulsar (i.e.
            spin-down luminosity) in erg/s. This is only used for the
            calculation of the irradiation efficiency so it does not
            enter in the modeling itself.
        read (bool): If True, Icarus will use the pre-calculated geodesic
            primitives. This is the recommended option, unless you have the
            pygts package installed to calculate it on the spot.
        
        Note: For an empirical approach, i.e. modelling the stellar
        temperature profile using spherical harmonics only, one can
        set tempgrav = 0., which disables the gravity darkening, and
        also set tirr = 0., which disables the irradiation.
        
        >>> fit = Photometry_temperature(atmo_fln, data_fln, ndiv, porb, x2sini, edot)
        """
        # Calling the parent class
        Photometry.__init__(self, atmo_fln, data_fln, ndiv, porb, x2sini, edot=edot, read=read)
        #self._Init_lightcurve(ndiv)
        
    def _Init_lightcurve(self, ndiv, read=True):
        """_Init_lightcurve(ndiv, read=True)
        Call the appropriate Lightcurve class and initialize
        the stellar array.
        
        >>> self._Init_lightcurve(ndiv)
        """
        self.star = Core.Star_temperature(ndiv)
        return

    def Make_surface(self, par, func_par=None, verbose=False):
        """Make_surface(par, func_par=None, verbose=False)
        This function gets the parameters to construct to companion
        surface model and calls the Make_surface function from the
        Lightcurve object.
        
        par: Parameter list.
            [0]: Orbital inclination in radians.
            [1]: Corotation factor.
            [2]: Roche-lobe filling.
            [3]: Companion temperature.
                Can be a float or a list of values to be used
                for the spherical harmonic composition of the temperature.
                Must have the form: [A_{00},A_{1-1},A_{10},A_{11}, ]
            [4]: Gravity darkening coefficient.
            [5]: K (projected velocity semi-amplitude) in m/s.
            [6]: Irradiation temperature.
                (lirr = eff * edot / (4*PI * a**2 * sigma))
                (tirr = (eff * edot / (4*PI * a**2 * sigma))**0.25)
                If None, will use self.tirr, otherwise, will use
                tirr and set self.tirr = tirr.
            [7]: Distance modulus (optional).
            [8]: Absorption A_V (optional).
            Note: Can also be a dictionary:
                par.keys() = ['aj','corotation','dm','filling','gravdark','incl','k1','tday','tnight']
        func_par (None): Function that takes the parameter vector and
            returns the parameter vector. This allow for possible constraints
            on the parameters. The vector returned by func_par must have a length
            equal to the number of expected parameters.
        
        Note: For an empirical approach, i.e. modelling the stellar
        temperature profile using spherical harmonics only, one can
        set tempgrav = 0., which disables the gravity darkening, and
        also set tirr = 0., which disables the irradiation.
        
        >>> Make_surface(par)
        """
        # Apply a function that can modify the value of parameters.
        if func_par is not None:
            par = func_par(par)
        # check if we are dealing with a dictionary
        if isinstance(par, dict):
            par = [par['incl'], par['corotation'], par['filling'], par['tnight'], par['gravdark'], par['k1'], par['tday'], par['dm'], par['aj']]
        
        # Verify parameter values to make sure they make sense.
        #if par[6] < par[3]: par[6] = par[3]
        # Let's move on with the flux calculation.
        q = par[5] * self.K_to_q
        
        if verbose:
            print( "#####\n" + str(par[0]) + ", " + str(par[1]) + ", " + str(par[2]) + ", " + str(par[3]) + ", " + str(par[4]) + ", " + str(par[5]) + ", " + str(par[6]) + ", " + str(par[7]) + ", " + str(par[8]) + "\n" + "q: " + str(q) )
        
        self.star.Make_surface(q=q, omega=par[1], filling=par[2], temp=par[3], tempgrav=par[4], tirr=par[6], porb=self.porb, k1=par[5], incl=par[0])
        return

######################## class Photometry_temperature ########################

