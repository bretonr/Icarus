# Licensed under a 3-clause BSD style license - see LICENSE

__all__ = ["Photometry_temperature"]

from ..Utils.import_modules import *
from .. import Core
from .Photometry import Photometry

logger = logging.getLogger(__name__)


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
    def __init__(self, atmo_fln, data_fln, ndiv, read=True, oldchi=False):
        """__init__(atmo_fln, data_fln, ndiv, read=True)
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
                Col 5: band calibration error, in magnitude
                Col 6: softening parameter for asinh magnitude conversion. If
                    the value is 0., then standard magnitudes are used.
                Col 7: flux or mag flag. Currently, all the data must be in the
                    same format.
                    'mag' means magnitude system
                    'flux' means flux system
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
                Col 5: band calibration error, in magnitude
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
                Col 5: band calibration error, in magnitude
                Col 6: filename
        ndiv (int): The number of surface element subdivisions. Defines how
            coarse/fine the surface grid is.
        read (bool): If True, Icarus will use the pre-calculated geodesic
            primitives. This is the recommended option, unless you have the
            pygts package installed to calculate it on the spot.
        
        Note: For an empirical approach, i.e. modelling the stellar
        temperature profile using spherical harmonics only, one can
        set tempgrav = 0., which disables the gravity darkening, and
        also set tirr = 0., which disables the irradiation.
        
        >>> fit = Photometry_temperature(atmo_fln, data_fln, ndiv, read=True)
        """
        # Calling the parent class
        Photometry.__init__(self, atmo_fln, data_fln, ndiv, read=read)
        #self._Init_lightcurve(ndiv)
        
    def _Init_lightcurve(self, ndiv, read=True, oldchi=False):
        """_Init_lightcurve(ndiv, read=True)
        Call the appropriate Lightcurve class and initialize
        the stellar array.
        
        >>> self._Init_lightcurve(ndiv)
        """
        logger.log(9, "start")
        self.star = Core.Star_temperature(ndiv)
        logger.log(9, "end")
        return

    def Make_surface(self, par, verbose=False):
        """
        This function gets the parameters to construct to companion
        surface model and calls the Make_surface function from the
        Lightcurve object.
        
        par: Parameter list.
            [0]: Mass ratio q = M2/M1, where M1 is the modelled star.
            [1]: Orbital period in seconds.
            [2]: Orbital inclination in radians.
            [3]: K1 (projected velocity semi-amplitude) in m/s.
            [4]: Corotation factor (Protation/Porbital).
            [5]: Roche-lobe filling in fraction of x_nose/L1.
            [6]: Gravity darkening coefficient.
                Should be 0.25 for radiation envelopes, 0.08 for convective.
            [7]: Star base temperature at the pole, before gravity darkening.
                This module uses a spherical harmonic temperature distribution.
                Can be a float or a list of values to be used for the spherical
                harmonic composition of the temperature.
                Must have the form: [A_{00},A_{1-1},A_{10},A_{11},...,A_{lm}]

            [8]: Irradiation temperature at the center of mass location.
                The effective temperature is calculated as T^4 = Tbase^4+Tirr^4
                and includes projection and distance effect.

            [3]: Companion temperature.

            Note: Can also be a dictionary:
                par.keys() = ['q','porb','incl','k1','omega','filling','tempgrav','temp','tirr']

        >>> self.Make_surface([10.,7200.,PIBYTWO,300e3,1.0,0.9,0.08,[4000.,100.,0.,0.],2000.])
        """
         ## check if we are dealing with a dictionary
        if isinstance(par, dict):
            self.star.Make_surface(
                q        = par['q'],
                porb     = par['porb'],
                incl     = par['incl'],
                k1       = par['k1'],
                omega    = par['omega'],
                filling  = par['filling'],
                tempgrav = par['tempgrav'],
                temp     = par['temp'],
                tirr     = par['tirr']
                )
        else:
            self.star.Make_surface(
                q        = par[0],
                porb     = par[1],
                incl     = par[2],
                k1       = par[3],
                omega    = par[4],
                filling  = par[5],
                tempgrav = par[6],
                temp     = par[7],
                tirr     = par[8]
                )

        if verbose:
            print( "Content on input parameter for Make_surface" )
            print( par )

        return

######################## class Photometry_temperature ########################

