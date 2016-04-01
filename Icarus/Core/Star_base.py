# Licensed under a 3-clause BSD style license - see LICENSE

__all__ = ["Star_base"]

from ..Utils.import_modules import *
from .. import Utils

logger = logging.getLogger(__name__)


######################## class Star_base ########################
class Star_base(object):
    """Star_base(object)
    This class allows to determine the flux of a star
    in a binary system using an atmosphere grid.
    
    The idea is the following:
        Some basic parameters of the binary system are provided (mass ratio, 
        corotation factor, filling factor, companion temperature, irradiated
        energy (spin down) from the primary). From these, the potential equation
        is solved and a grid of [temperature - surface gravity] is calculated.
        Then, provided an orbital inclination and an orbital phase, the cos(emission
        angle to observer) is calculated. For each (T, g, mu) triplet, we can infer
        the radiance from an atmosphere grid, which, after multiplicating by the
        surface element of that it covers, provides the contribution to the total
        luminosity of the star. It is taken into account that some energy received
        from the companion heats up the exposed side of the star.
    """
    def __init__(self, ndiv, atmo_grid=None):
        """__init__
        Initialize the class instance.
        
        ndiv: The number of surface element subdivisions. Defines how
            coarse/fine the surface grid is.
        atmo_grid (None): An atmosphere model grid from which the
            radiance is interpolated.
        
        It is optional to provide an atmosphere grid. If none is provided, it will
        have to be passed as a parameter to the routine calculating the flux.
        
        >>> star = Star_base(nafl)
        """
        logger.log(9, "start")
        # We define some useful quantities.
        # We set the class attributes
        if atmo_grid is not None:
           self.atmo_grid = atmo_grid 
        self.ndiv = ndiv
        # Instantiating some class attributes.
        self.q = None
        self.omega = None
        self.filling = None
        self.temp = None
        self.tempgrav = None
        self.tirr = None
        self.porb = None
        self.k1 = None
        self.incl = None
        logger.log(9, "end")

    def _Area(self, arl, r):
        """_Area(arl, r)
        Returns the surface area given a solid angle and radius
        arl: solid angle
        r: radius
        
        >>> self._Area(arl, r)
        area
        """
        return arl*r**2

    def Bbody_flux(self, phase, limbdark, atmo_grid=None, proj=None):
        """Bbody_flux(phase, limbdark, atmo_grid=None)
        Returns the blackbody flux of a star which is modulated
        by a linear limb darkening coefficient (i.e. some linear
        fall-off of the radiance vs. mu).
        
        phase: orbital phase (in orbital fraction; 0: companion 
            in front, 0.5: companion behind).
        limbdark: limb darkening coefficient.
        atmo_grid (optional): atmosphere grid instance to 
            calculate the flux.
        proj (optional): projection effect to scale the flux to real flux
            units. If None is provided, will call _Proj with the current
            orbital separation as input parameter.
        
        >>> self.Bbody_flux(phase, limbdark)
        bbody_flux
        """
        if atmo_grid is None:
            atmo_grid = self.atmo_grid
        if proj is None:
            proj = self._Proj(self.separation)
        const1 = 3.74177e-5
        const2 = 1.43877
        mu = self._Mu(phase)
        limbnorm = 1.-limbdark/3.
        limb0 = 1/limbnorm
        limb1 = limbdark/limbnorm
        bbconst = const1/(C*100)/cts.PI/atmo_grid.wav**3
        f = bbconst/(np.exp(const2/self.temp/atmo_grid.grid_lam)-1)
        if isinstance(f, np.ndarray):
            f.shape = f.size,1
        inds = mu > 0
        return (self.area[inds] * f * mu[inds] * (limb0-limb1*(1-mu[inds]))).sum(axis=-1) * proj

    def Bol_flux(self, phase, proj=None):
        """Bol_flux(phase)
        Returns the bolometric flux of a star
        (i.e. mu*sigma*T**4).
        
        phase: orbital phase (in orbital fraction; 0: companion 
            in front, 0.5: companion behind).
        proj (optional): projection effect to scale the flux to real flux
            units. If None is provided, will call _Proj with the current
            orbital separation as input parameter.
        
        >>> self.Bol_flux(phase)
        bol_flux
        """
        if proj is None:
            proj = self._Proj(self.separation)
        sigmabypi = 5.67051e-5/cts.PI # sigma/pi (sigma is the stefan-boltzman constant)
        mu = self._Mu(phase)
        inds = mu > 0
        return (self.area[inds] * (mu[inds]*sigmabypi*self.temp**4)).sum() * proj

    def _Calc_qp1by2om2(self):
        """_Calc_qp1by2om2
        Sets the qp1by2om2 parameter.
        
        >>> self._Calc_qp1by2om2()
        """
        self.qp1by2om2 = (self.q+1.)/2.*self.omega**2
        return

    def _Calc_teff(self, temp=None, tirr=None):
        """_Calc_teff(temp=None, tirr=None)
        Calculates the log of the effective temperature on the
        various surface elements of the stellar surface grid.
        
        For surface elements that are not exposed to the primary's
        irradiated flux, the base temperature profile is described
        using a constant temperature. This temperature is then 
        modified by a factor that takes into account the gravity 
        darking through the 'tempgrav' exponent specified when 
        calling Make_surface().
        
        For the exposed surface elements, an irradiation temperature
        is added the base temperature as:
            (Tbase**4 + coschi/r**2*Tirr**4)**0.24
        where 'coschi' is the angle between the normal to the surface
        element and the direction to the irradiation source and 'r'
        is the distance.
        
        temp (None): Base temperature of the star.
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
        # We calculate the gravity darkening correction to the temperatures across the surface and multiply them by the base temperature.
        teff = self.temp*self._Gravdark()
        # We apply the irradiation to the surface visible to the irradiation source.
        inds = self.coschi > 0.0
        if inds.any() and self.tirr != 0.:
            teff[inds] = (teff[inds]**4+self.coschi[inds]*self.tirr**4/self.rx[inds]**2)**0.25
        if (teff <= 0).any():
            print( self.temp.min() )
            print( self.temp.max() )
            print( self.tirr )
            print( teff.min() )
            print( teff.max() )
            print( (teff <= 0).sum() )
            print( teff[teff <=0 ] )
        self.logteff = np.log(teff)
        return

    def Doppler_boosting(self, logteff, logg):
        """ Doppler_boosting(logteff, logg)
        Returns the Doppler boosting factor for the values of logteff and logg.
        Currently does a bilinear interpolation for a few values.
        #         logg=4.0  logg=3.5
        # teff=6250 K  3.618     3.591
        # teff=6500 K  3.460     3.428
        
        logteff: log of temperature.
        logg: log of surface gravity.
        """
        try:
            if logteff[0] < np.log(10000.):
                case = 1
            else:
                case = 2
        except:
            print( "Problem with the test for the Doppler boosting range!!!" )
            return 0.
        
        if case == 1:
            logg_vec = np.r_[3.5, 4.0]
            logteff_vec = np.log(np.r_[6250., 6500.])
            lookup = np.array([[3.591, 3.428],[3.618, 3.460]])
        elif case == 2:
            logg_vec = np.r_[6.0, 7.0]
            logteff_vec = np.log(np.r_[14000., 15000.])
            lookup = np.array([[1.93, 1.87],[2.05, 1.98]])
        else:
            print( "Problem with the test for the Doppler boosting range!!!" )
            return 0.
        
        #lookup.shape = logg_vec.size, logteff_vec.size
        ### Here we use a shortcut by taking the average
        w_logg, j_logg = Utils.Series.Getaxispos_scalar(logg_vec, logg.mean())
        w_logteff, j_logteff = Utils.Series.Getaxispos_scalar(logteff_vec, logteff.mean())
        ### here we use the full blown version
        #w_logg, j_logg = Utils.Series.Getaxispos_vector(logg_vec, logg)
        #w_logteff, j_logteff = Utils.Series.Getaxispos_vector(logteff_vec, logteff)
        doppler = (1-w_logg) * ( (1-w_logteff)*lookup[j_logg,j_logteff] + w_logteff*lookup[j_logg,1+j_logteff] ) + w_logg * ( (1-w_logteff)*lookup[1+j_logg,j_logteff] + w_logteff*lookup[1+j_logg,1+j_logteff] )
        return doppler
    
    def Doppler_boosting_old(self, logteff, logg):
        """ Doppler_boosting_old(logteff, logg)
        Returns the Doppler boosting factor for the values of logteff and logg.
        Currently does a bilinear interpolation for a few values.
        #         logg=4.0  logg=3.5
        # teff=6250 K  3.618     3.591
        # teff=6500 K  3.460     3.428
        
        logteff: log of temperature.
        logg: log of surface gravity.
        """
        case = 0
        try:
            if logteff[0] < 10000.:
                case = 1
            else:
                case = 2
        except:
            try:
                if logteff < 10000.:
                    case = 1
                else:
                    case = 2
            except:
                case = 0
        if case == 1:
            x1 = 3.5
            x2 = 4.0
            y1 = np.log(6250.)
            y2 = np.log(6500.)
            Q11 = 3.591
            Q12 = 3.428
            Q21 = 3.618
            Q22 = 3.460
        elif case == 2:
            x1 = 6.0
            x2 = 7.0
            y1 = np.log(14000.)
            y2 = np.log(15000.)
            Q11 = 1.93
            Q12 = 1.87
            Q21 = 2.05
            Q22 = 1.98
        else:
            print( "Problem with the test for the Doppler boosting range!!!" )
            return 0.
        doppler = ( (Q11*(x2-logg) + Q21*(logg-x1))*(y2-logteff) + (Q12*(x2-logg) + Q22*(logg-x1))*(logteff-y1) ) / ((x2-x1)*(y2-y1))
        return doppler

    def Filling(self):
        """Filling()
        Returns the volume-averaged filling factor of the star
        in units Roche lobe radius.
        
        >>> self.Filling()
        """
        filling = self.filling
        self.Make_surface(filling=1.)
        radius_RL = self.Radius()
        self.Make_surface(filling=filling)
        radius = self.Radius()
        return radius/radius_RL

    def Flux(self, phase, atmo_grid=None, gravscale=None, proj=None, nosum=False, details=False, mu=None, inds=None):
        """
        Return the flux interpolated from the atmosphere grid.

        phase: orbital phase (in orbital fraction; 0: companion 
            in front, 0.5: companion behind).
        atmo_grid (optional): atmosphere grid instance used to
            calculate the flux.
        gravscale (optional): gravitational scaling parameter.
        proj (optional): projection effect to scale the flux to real flux
            units. If None is provided, will call _Proj with the current
            orbital separation as input parameter.
        nosum (False): if true, will no sum across the surface.
        details (False): if true, will return (flux, Keff, vsini, Teff).
        mu (None): if provided, the vector of mu angles (angle between
            line of sight and surface normal).
        inds (None): if provided, the list of indices to use for
            the flux calculation. Can be handy to approximate
            eclipses.
        
        >>> self.Flux(phase)
        flux
        """
        logger.log(9, "start")
        if atmo_grid is None:
            atmo_grid = self.atmo_grid
        if gravscale is None:
            gravscale = self._Gravscale()
        if proj is None:
            proj = self._Proj(self.separation)
        if mu is None:
            mu = self._Mu(phase)
        if inds is None:
            inds = mu > 0
        
        logteff = self.logteff[inds]
        logg = self.logg[inds]+gravscale
        mu = mu[inds]
        area = self.area[inds]
    
        if details:
            v = self._Velocity_surface(phase)[inds]
            fsum, Keff, vsini, Teff = atmo_grid.Get_flux_details(logteff, logg, mu, area, v)
            if proj != 1:
                fsum *= proj
            return fsum, Keff*cts.c, vsini*cts.c, Teff
        elif nosum:
            fsum = atmo_grid.Get_flux_nosum(logteff, logg, mu, area)
            if proj != 1:
                fsum *= proj
            return fsum
        else:
            fsum = atmo_grid.Get_flux(logteff, logg, mu, area)
            if proj != 1:
                fsum *= proj
            return fsum

        logger.log(9, "end")
        return

    def Flux_doppler(self, phase, atmo_grid=None, gravscale=None, proj=None, nosum=False, mu=None, inds=None, velocity=0., atmo_doppler=None):
        """
        Return the flux interpolated from the atmosphere grid.
        Takes into account the Doppler shift of the different surface
        elements due to the orbital velocity.
        
        phase: orbital phase (in orbital fraction; 0: companion 
            in front, 0.5: companion behind).
        atmo_grid (optional): atmosphere grid instance used to
            calculate the flux.
        gravscale (optional): gravitational scaling parameter.
        proj (optional): projection effect to scale the flux to real flux
            units. If None is provided, will call _Proj with the current
            orbital separation as input parameter.
        nosum (False): if true, will no sum across the surface.
        mu (None): if provided, the vector of mu angles (angle between
            line of sight and surface normal).
        inds (None): if provided, the list of indices to use for
            the flux calculation. Can be handy to approximate
            eclipses.
        velocity (optional): extra velocity in m/s to be added.
        atmo_doppler (optional): AtmoGridDoppler instance containing a grid of Doppler
            boosting factors. Must be the same dimensions as the atmosphere grid.
            This is needed for the photometry atmosphere grid, but not for the
            spectroscopy.
        debug (optional): whether to print extra debugging information or not

        >>> self.Flux_doppler(phase)
        flux
        """
        logger.log(9, "start")
        if atmo_grid is None:
            atmo_grid = self.atmo_grid
        if gravscale is None:
            gravscale = self._Gravscale()
        if proj is None:
            proj = self._Proj(self.separation)
        if mu is None:
            mu = self._Mu(phase)
        if inds is None:
            inds = mu > 0

        v = self._Velocity_surface(phase, velocity=velocity)

        if atmo_doppler is not None:
            if nosum:
                fsum = atmo_grid.Get_flux_doppler_nosum(self.logteff[inds], self.logg[inds]+gravscale, mu[inds], self.area[inds], v[inds], atmo_doppler)
            else:
                fsum = atmo_grid.Get_flux_doppler(self.logteff[inds], self.logg[inds]+gravscale, mu[inds], self.area[inds], v[inds], atmo_doppler)
        else:
            if nosum:
                fsum = atmo_grid.Get_flux_doppler_nosum(self.logteff[inds], self.logg[inds]+gravscale, mu[inds], self.area[inds], v[inds])
            else:
                logger.log(5, '-'*20)
                logger.log(5, 'logteff yo')
                logger.log(5, self.logteff[inds])
                logger.log(5, '-'*20)
                logger.log(5, 'logg')
                logger.log(5, self.logg[inds]+gravscale)
                logger.log(5, '-'*20)
                logger.log(5, 'mu')
                logger.log(5, mu[inds])
                logger.log(5, '-'*20)
                logger.log(5, 'area')
                logger.log(5, self.area[inds])
                logger.log(5, '-'*20)
                logger.log(5, 'v')
                logger.log(5, v[inds])
                logger.log(5, '-'*20)
                fsum = atmo_grid.Get_flux_doppler(self.logteff[inds], self.logg[inds]+gravscale, mu[inds], self.area[inds], v[inds])

        if proj != 1:
            fsum *= proj
        logger.log(9, "stop")
        return fsum

    def _Geff(self, dpsidx, dpsidy, dpsidz):
        """_Geff(dpsidx, dpsidy, dpsidz)
        Returns the effective gravity at a given point having
        element surface gravity.
        
        dpsidx, dpsidy, dpsidz: vector elements of surface gravity
        
        >>> self._Geff(dpsidx, dpsidy, dpsidz)
        geff
        """
        return np.sqrt(dpsidx**2+dpsidy**2+dpsidz**2)

    def _Gravdark(self):
        """_Gravdark()
        Returns the companion temperature after applying
        the gravity darkening coefficient.
        
        >>> self._Gravdark()
        tcorr
        """
        return np.exp(self.tempgrav*np.log(10.)*(self.logg-self.logg_pole))

    def _Gravscale(self):
        """_Gravscale()
        Returns the gravitational scaling. This is a quantity added
        to the log(g) value and provided a real physical value from
        the companion mass and orbital separation.
        gravscale = log10(gmsun * M1 / r**2)
        gmsun = G*Msun (in cgs)
        M1 = mass in Msun
        r = orbital separation in cm
        
        >>> self._Gravscale()
        gravscale
        """
        gmsun = 1.3271243999e26
        # Convert the orbital separation from meters to centimeters
        return np.log10(gmsun*self.mass1/(self.separation*100)**2)

    def Keff(self, phase, gravscale=None, atmo_grid=None):
        """Keff(phase, gravscale=None, atmo_grid=None)
        Return the effective velocity of the star in m/s (i.e. averaged over
        the visible surface and flux intensity weighted).
        
        phase: orbital phase (in orbital fraction; 0: companion 
            in front, 0.5: companion behind).
        gravscale (optional): gravitational scaling parameter.
        atmo_grid (optional): atmosphere grid instance used to
            calculate the flux.
        
        >>> self.Keff(phase)
        Keff
        """
        if atmo_grid is None:
            atmo_grid = self.atmo_grid
        if gravscale is None:
            gravscale = self._Gravscale()
        mu = self._Mu(phase)
        v = self._Velocity_surface(phase)
        inds = (mu > 0).nonzero()[0]
        fsum, Keff = atmo_grid.Get_flux_Keff(self.logteff[inds],self.logg[inds]+gravscale,mu[inds],self.area[inds],v[inds])
        return Keff*cts.c

    def Mag_bbody_flux(self, phase, limbdark, proj=None, atmo_grid=None):
        """Mag_flux(phase)
        Returns the blackbody magnitude of a star.
        
        phase: orbital phase (in orbital fraction; 0: companion 
            in front, 0.5: companion behind).
        limbdark: limb darkening coefficient.
        proj (optional): projection effect to scale the flux to real flux
            units. If None is provided, will call _Proj with the current
            orbital separation as input parameter.
        atmo_grid (optional): atmosphere grid instance to calculate the flux.
        
        >>> self.Mag_bbody_flux(phase, limbdark, a=None)
        mag_bbody_flux
        """
        if atmo_grid is None:
            atmo_grid = self.atmo_grid
        if proj is None:
            proj = self._Proj(self.separation)

        return -2.5*np.log10(self.Bbody_flux(phase, limbdark, atmo_grid=atmo_grid, proj=proj)) + atmo_grid.meta['zp']

    def Mag_bol_flux(self, phase, proj=None):
        """Mag_bol_flux(phase, a=None)
        Returns the bolometric magnitude of a star.
        
        phase: orbital phase (in orbital fraction; 0: companion 
            in front, 0.5: companion behind).
        proj (optional): projection effect to scale the flux to real flux
            units. If None is provided, will call _Proj with the current
            orbital separation as input parameter.
        
        >>> self.Mag_bol_flux(phase)
        mag_bol_flux
        """
        bolflux0 = 2.54e-5 # bolometric zero-point flux
        if proj is None:
            proj = self._Proj(self.separation)
        return -2.5*np.log10(self.Bol_flux(phase, proj=proj)/bolflux0)

    def Mag_flux(self, phase, gravscale=None, proj=None, atmo_grid=None):
        """Mag_flux(phase, gravscale=None, a=None, atmo_grid=None)
        Returns the magnitude interpolated from the atmosphere grid.
        
        phase: orbital phase (in orbital fraction; 0: companion 
            in front, 0.5: companion behind).
        gravscale (optional): gravitational scaling parameter.
        proj (optional): projection effect to scale the flux to real flux
            units. If None is provided, will call _Proj with the current
            orbital separation as input parameter.
        atmo_grid (optional): atmosphere grid instance to work from to 
            calculate the flux.
        
        >>> self.Mag_flux(phase)
        mag_flux
        """
        if atmo_grid is None:
            atmo_grid = self.atmo_grid
        if proj is None:
            proj = self._Proj(self.separation)
        if gravscale is None:
            gravscale = self._Gravscale()
        return -2.5*np.log10(self.Flux(phase, gravscale=gravscale, proj=proj, atmo_grid=atmo_grid)) + atmo_grid.meta['zp']

    def Mag_flux_doppler(self, phase, gravscale=None, proj=None, atmo_grid=None, velocity=0., atmo_doppler=None):
        """
        Returns the magnitude interpolated from the atmosphere grid.
        Takes into account the Doppler shift of the different surface
        elements due to the orbital velocity.
        
        phase: orbital phase (in orbital fraction; 0: companion 
            in front, 0.5: companion behind).
        gravscale (optional): gravitational scaling parameter.
        proj (optional): projection effect to scale the flux to real flux
            units. If None is provided, will call _Proj with the current
            orbital separation as input parameter.
        atmo_grid (optional): atmosphere grid instance to work from to 
            calculate the flux.
        velocity (optional): extra velocity in m/s to be added.
        atmo_doppler (optional): AtmoGridDoppler instance containing a grid of Doppler
            boosting factors. Must be the same dimensions as the atmosphere grid.
            This is needed for the photometry atmosphere grid, but not for the
            spectroscopy.
        
        >>> self.Mag_flux_doppler(phase)
        mag_flux_doppler
        """
        if atmo_grid is None:
            atmo_grid = self.atmo_grid
        if proj is None:
            proj = self._Proj(self.separation)
        if gravscale is None:
            gravscale = self._Gravscale()
        return -2.5*np.log10(self.Flux_doppler(phase, gravscale=gravscale, proj=proj, atmo_grid=atmo_grid, velocity=velocity, atmo_doppler=atmo_doppler)) + atmo_grid.meta['zp']

    def Make_surface(self, q=None, omega=None, filling=None, temp=None, tempgrav=None, tirr=None, porb=None, k1=None, incl=None):
        """Make_surface(q=None, omega=None, filling=None, temp=None, tempgrav=None, tirr=None, porb=None, k1=None, incl=None)
        Provided some basic parameters about the binary system,
        calculates the surface grid values for surface gravity,
        temperature, surface area, etc.
        
        This function is just a conveniance function that calls
        internal functions. If q, omega and filling are not changed,
        only the temperature is recalculated.
        
        q (None): mass ratio (M2/M1). M1 is the star being modelled.
        omega (None): co-rotation factor (Protation/Porbital).
        filling (None): Roche-lobe filling factor (x_nose/L1).
        temp (None): surface temperature of the star.
            The temperature is defined at the pole of the star, before 
            irradiation and gravity darkening.
        tempgrav (None): gravity darkening coefficient.
            The gravity darkening is accounted as follow:
                T = Tpole * (g/g_pole)**tempgrav
            Should be 0.25 for radiation envelopes, 0.08 for convective.
        tirr (None): irradiation temperature.
            (lirr = eff * edot / (4*PI * a**2 * sigma))
            (tirr = (eff * edot / (4*PI * a**2 * sigma))**0.25)
            The irradiation temperature is added as
                Teff**4 = tirr**4 + temp**4.
            The 1/distance**2 effect is accounted for. The fiducial reference
            point is the center of mass of the star.
            The projection effect is also factored in.
        porb (None): orbital period, in seconds.
        k1 (None): velocity semi-amplitude, in m/s.
        incl (None): orbital inclination, in radians.
        
        >>> self.Make_surface(q, omega, filling, temp, tempgrav, tirr, porb, k1, incl)
        """
        logger.log(9, "start")
        #print 'Begin Make_surface'
        #print q, omega, filling, temp, tempgrav, tirr
        redo_surface = False
        redo_teff = False
        redo_orbital = False
        # We go through the list of optional parameters in order
        # to determine was has to be recalculated
        if q is not None:
            if q != self.q:
                self.q = q
                redo_surface = True
                redo_teff = True
                redo_orbital = True
        if omega is not None:
            if omega != self.omega:
                self.omega = omega
                redo_surface = True
                redo_teff = True
        if filling is not None:
            if filling != self.filling:
                self.filling = filling
                redo_surface = True
                redo_teff = True
        if temp is not None:
            temp = np.atleast_1d(temp)
            if np.any(temp != self.temp):
                self.temp = temp
                redo_teff = True
        if tempgrav is not None:
            if tempgrav != self.tempgrav:
                self.tempgrav = tempgrav
                redo_teff = True
        if tirr is not None:
            if tirr != self.tirr:
                self.tirr = tirr
                redo_teff = True
        if porb is not None:
            if porb != self.porb:
                self.porb = porb
                redo_orbital = True
        if k1 is not None:
            if k1 != self.k1:
                self.k1 = k1
                redo_orbital = True
        if incl is not None:
            if incl != self.incl:
                self.incl = incl
                redo_orbital = True
        if redo_surface:
            #print 'Going to _Surface()'
            self._Surface()
        if redo_teff:
            #print 'Going to _Calc_teff()'
            self._Calc_teff()
        if redo_orbital:
            #print 'Going to _Orbital_parameters()'
            self._Orbital_parameters()
        #print 'End Make_surface'
        logger.log(9, "end")
        return

    def _Mu(self, phase):
        """_Mu(phase)
        Returns the cos(angle) of the emission angle with respect
        to the observer.
        
        phase: orbital phase (in orbital fraction; 0: companion 
            in front, 0.5: companion behind).
        
        >>> self._Mu(phase)
        mu
        """
        return -np.sin(self.incl)*(np.cos(cts.TWOPI*phase)*self.gradx+np.sin(cts.TWOPI*phase)*self.grady)+np.cos(self.incl)*self.gradz

    def _Orbital_parameters(self):
        """_Orbital_parameters()
        This function uses the class variables q, k1, porb
        and incl in order to derive additional useful quantities
        such as a1sini, a1, d1, mass1, mass2.
        
        >>> self._Orbital_parameters()
        """
        # We calculate the projected semi-major axis
        self.a1sini = self.k1 * self.porb / cts.TWOPI
        self.a1 = self.a1sini / np.sin(self.incl)
        # We calculate the orbital separation
        # Note: (1+q)/q is the ratio of the separation to the semi-major axis
        self.separation = self.a1 * (1+self.q)/self.q
        # We calculate the mass of the primary using Kepler's 3rd law and convert to Msun
        # Law: (Porb/cts.TWOPI)**2 = a**3/G/(M+m)
        self.mass1 = (self.separation**3 / (cts.G * (self.porb/cts.TWOPI)**2 * (1+self.q))) / cts.Msun
        self.mass2 = self.q*self.mass1
        return

    def _Potential(self, x, y, z):
        """_Potential(x, y, z)
        Calculates the potential at a position (x, y, z).
        
        x, y, z: position (can be vectors or scalars)
        
        >>> self._Potential(x, y, z)
        rc, rx, dpsi, dpsidx, dpsidy, dpsidz, psi
        """
        logger.log(9, "start")
        rc, rx, dpsi, dpsidx, dpsidy, dpsidz, psi = Utils.Binary.Potential(x, y, z, self.q, self.qp1by2om2)
        logger.log(9, "end")
        return rc, rx, dpsi, dpsidx, dpsidy, dpsidz, psi

    def _Proj(self, r):
        """_Proj(r)
        Returns the projection parameter (i.e. 1/r**2 fall-off of luminosity)
        
        r: orbital separation in meters.
        
        >>> self._Proj(r)
        proj
        """
        parsec = 3.085678e18 # parsec in cm
        # the factor 100 is to convert from m to cm
        return ((r*100)/10./parsec)**2

    def _Radius(self, cosx, cosy, cosz, psi0, rtry):
        """_Radius(cosx, cosy, cosz, psi0, rtry)
        Determines the radius of the star at a given angular position.
        If cosx,cosy,cosz are vectors, will return a vector of radii.
        
        cosx, cosy, cosz: angular position (scalar or vector)
        psi0: gravitational potential of the star (scalar)
        rtry: guess radius (scalar)
        
        >>> self._Radius(cosx, cosy, cosz, psi0, rtry)
        radius
        """
        logger.log(9, "start")
        if isinstance(cosx, np.ndarray):
            radius = Utils.Binary.Radii(cosx, cosy, cosz, psi0, rtry, self.q, self.qp1by2om2)
        else:
            radius = Utils.Binary.Radius(cosx, cosy, cosz, psi0, rtry, self.q, self.qp1by2om2)
        logger.log(9, "end")
        return radius

    def _Radius_slow(self, cosx, cosy, cosz, psi0, rtry):
        """_Radius_slow(cosx, cosy, cosz, psi0, rtry)
        Determines the radius of the star at a given angular position.
        This is the slow, "well tested" way.
        
        cosx, cosy, cosz: angular position
        psi0: nominal potential ???
        rtry: guess radius
        
        >>> self._Radius_slow(cosx, cosy, cosz, psi0, rtry)
        radius
        """
        def get_radius(r, cosx, cosy, cosz):
            rc, rx, dpsi, dpsidx, dpsidy, dpsidz, psi = self._Potential(r*cosx,r*cosy,r*cosz)
            dpsidr = dpsidx*cosx+dpsidy*cosy+dpsidz*cosz
            dr = -(psi-psi0)/dpsidr
            return dr
        if isinstance(cosx, (list,tuple,np.ndarray)):
            try:
                radius = np.array([scipy.optimize.newton(get_radius, rtry, args=(cosx[i], cosy[i], cosz[i])) for i in np.arange(cosx.size)])
            except:
                print "The cosx, cosy and cosz arrays don't have the same length!"
                radius = 0.
        else:
            radius = scipy.optimize.newton(get_radius, rtry, args=(cosx, cosy, cosz))
        return radius

    def Radius(self):
        """Radius()
        Returns the volume-averaged radius of the star, in
        units of orbital separation.
        
        >>> self.Radius()
        """
        logger.log(9, "start")
        sindeltalfby2 = np.sin(cts.PIBYTWO/self.ndiv)
        solidangle = []
        solidangle.append(cts.TWOPI*(1.-np.sqrt(1.-sindeltalfby2**2)))
        solidangle_nbet = 4*cts.PI*np.sin(cts.PI*np.arange(1,self.ndiv)/self.ndiv)*sindeltalfby2/self.nbet
        [solidangle.extend(s.repeat(i)) for s,i in zip(solidangle_nbet,self.nbet)]
        solidangle.append(cts.TWOPI*(1.-np.sqrt(1.-sindeltalfby2**2)))
        vol = (self.rc**3*np.array(solidangle)).sum()/3
        logger.log(9, "end")
        return (vol/(4*cts.PI/3))**(1./3.)

    def Roche(self):
        """Roche()
        Returns the volume-averaged Roche lobe radius
        of the star in units of orbital separation.
        
        >>> self.Roche()
        """
        logger.log(9, "start")
        filling = self.filling
        self.Make_surface(filling=1.)
        radius = self.Radius()
        self.Make_surface(filling=filling)
        logger.log(9, "end")
        return radius

    def _Saddle(self, xtry):
        """_Saddle(xtry)
        Returns the saddle point given an guess position
        
        xtry: guess position
        
        >>> self._Saddle(0.5)
        saddle
        """
        logger.log(9, "call")
        return Utils.Binary.Saddle(xtry, self.q, self.qp1by2om2)

    def _Saddle_old(self, xtry):
        """_Saddle_old(xtry)
        Returns the saddle point given an guess position
        This version does not always converge.
        
        xtry: guess position
        
        >>> self._Saddle_old(0.5)
        saddle
        """
        def get_saddle(x):
            rc, rx, dpsi, dpsidx, dpsidy, dpsidz, psi = self._Potential(x, 0., 0.)
            d2psidx2 = dpsi+3.*(x**2/rc**5+self.q*(x-1.)**2/rx**5)+2.*self.qp1by2om2
            dx = -dpsidx/d2psidx2
            return dx/(x+dx)
        saddle = scipy.optimize.newton(get_saddle, xtry)
        return saddle

    def _Surface(self):
        """_Surface()
        Calculates the surface grid values of surface gravity
        and surface element area by solving the potential
        equation.
        
        >>> self._Surface()
        """
        logger.log(9, "start")
#        print( "Begin _Surface()" )
        # Calculate some quantities
        self._Calc_qp1by2om2()
        sindeltalfby2 = np.sin(cts.PIBYTWO/self.ndiv)
        arl1 = cts.TWOPI*(1.-np.sqrt(1.-sindeltalfby2**2)) # solid angle
        
#        print 'Start surface'
        # Calculate the initial saddle point
#        print 'Saddle'
        xl1 = self._Saddle(0.5)
        self.L1 = xl1
        self.rc_l1 = self.filling*xl1
        # Calculate the potential at the saddle point
#        print 'Potential psil1'
        psil1 = self._Potential(xl1, 0., 0.)[-1]
#        print 'xl1,psil1 '+str(xl1)+' '+str(psil1)
        # Calculate the potential at rc_l1
#        print 'Potential psi0'
        trc, trx, dpsi, dpsidx, dpsidy, dpsidz, psi0 = self._Potential(self.rc_l1, 0., 0.)
#        print 'rc_l1,psi0 '+str(self.rc_l1)+' '+str(psi0)
#        print 'psi0 = ', psi0
#        print 'psil1 = ', psil1
#        print 'xl1 = ', xl1
#        print 'rc_l1 = ', self.rc_l1
#        print 'trc: %f, trx %f, dpsi %f, dpsidx %f, dpsidy %f, dpsidz %f, psi0 %f' % (trc, trx, dpsi, dpsidx, dpsidy, dpsidz, psi0)
        # Store the cos values
        cosx = []
        cosy = []
        cosz = []
        cosx.append(1.)
        cosy.append(0.)
        cosz.append(0.)
        # Store the rx and rx values
        # rc corresponds to r1 from Tjemkes et al., the distance from the center of mass of the pulsar companion
        rc = []
        # rx corresponds to r2, the distance from the center of mass of the pulsar
        rx = []
        coschi = []
        rc.append(trc)
        rx.append(trx)
        coschi.append(1.)
        # Calculate surface gravity        
        geff = self._Geff(dpsidx, dpsidy, dpsidz)
#        print 'geff '+str(geff)
        logg = []
        area = []
        logg.append(np.log10(geff))
        area.append(self._Area(arl1, trc))
#        print 'area'+str(area)
        # Calculate surface gradient components
        gradx = []
        grady = []
        gradz = []
        gradx.append(-dpsidx/geff)
        grady.append(-dpsidy/geff)
        gradz.append(-dpsidz/geff)
        #  rl180 is the Roche-lobe radius on the far side
#        print 'about to calculate rl180'
        try:
            rl180 = self._Radius(-1.,0.,0.,psil1,0.9*xl1)
        except:
            rl180 = self._Radius(-1.,0.,0.,psil1,0.1*xl1)
#        print 'rl180 '+str(rl180)
        
#        print 'Start surface loop'
        # Calculate useful quantities for each slice of the surface
        tcosx = np.cos(cts.PI*np.arange(1,self.ndiv)/self.ndiv)
        tsinx = np.sqrt(1.-tcosx**2)
        #rl = self._Radius(tcosx, tsinx, np.zeros(self.ndiv, dtype=float), psil1, np.repeat(rl180,self.ndiv-1))
#        print 'about to calculate rl'
        rl = self._Radius(tcosx, tsinx, np.zeros(self.ndiv, dtype=float), psil1, rl180)
#        print 'rl '#+str(rl)
        rtry = self._Radius(-1., 0., 0., psi0, rl180)
#        print 'rtry '+str(rtry)
        ar = 4*cts.PI*tsinx*sindeltalfby2
        nbet = np.round(ar/arl1*(rl/rl180)**2).astype(int)
        ar = ar/nbet
        # Define a function to calculate the quantities for each slice
        def get_slice(tcosx, tsinx, nbet, ar):
            tcosx = np.resize(tcosx,nbet)
            bet = cts.TWOPI*(np.arange(1,nbet+1)-0.5)/nbet
            tcosy = tsinx*np.cos(bet)
            tcosz = tsinx*np.sin(bet)
            #r = self._Radius(tcosx, tcosy, tcosz, psi0, np.repeat(rl180,nbet))
            r = self._Radius(tcosx, tcosy, tcosz, psi0, rtry)
            trc, trx, dpsi, dpsidx, dpsidy, dpsidz, psi = self._Potential(r*tcosx,r*tcosy,r*tcosz)
            geff = self._Geff(dpsidx, dpsidy, dpsidz)
            cosx.extend(tcosx)
            cosy.extend(tcosy)
            cosz.extend(tcosz)
            coschi.extend(-(trc*(tcosx**2+tcosy**2+tcosz**2)-tcosx)/trx)
            rc.extend(r)
            rx.extend(trx)
            area.extend(self._Area(ar*geff/(-dpsidx*tcosx-dpsidy*tcosy-dpsidz*tcosz), r))
            logg.extend(np.log10(geff))
            gradx.extend(-dpsidx/geff)
            grady.extend(-dpsidy/geff)
            gradz.extend(-dpsidz/geff)
        # Iterate through the slices
        [get_slice(itcosx,itsinx,inbet,iar) for itcosx,itsinx,inbet,iar in zip(tcosx,tsinx,nbet,ar)]

        # Calculate the final values
        r = self._Radius(-1.,0.,0.,psi0,rl180)
        trc, trx, dpsi, dpsidx, dpsidy, dpsidz, psi = self._Potential(-r,0.,0.)
        geff = self._Geff(dpsidx, dpsidy, dpsidz)
        cosx.append(-1.)
        cosy.append(0.)
        cosz.append(0.)
        coschi.append(-1.)
        rc.append(r)
        rx.append(trx)
        area.append(self._Area(arl1, trc))
        logg.append(np.log10(geff))
        gradx.append(-dpsidx/geff)
        grady.append(-dpsidy/geff)
        gradz.append(-dpsidz/geff)
        # rc_pole is the Roche-lobe radius at 90 degrees, i.e. perpendicular to the line separating the two stars
#        print 'psi0,r '+str(psi0)+' '+str(r)
        rc_pole = self._Radius(0.,0.,1.,psi0,r)
        trc, trx, dpsi, dpsidx, dpsidy, dpsidz, psi = self._Potential(0.,0.,rc_pole)
        self.logg_pole = np.log10(np.sqrt(dpsidx**2+dpsidy**2+dpsidz**2))

        rc_eq = self._Radius(0.,1.,0.,psi0,r)
        trc, trx, dpsi, dpsidx, dpsidy, dpsidz, psi = self._Potential(0.,rc_eq,0.)
        self.logg_eq = np.log10(np.sqrt(dpsidx**2+dpsidy**2+dpsidz**2))

#        print 'rc_pole,dpsidx,dpsidy,dpsidz '+str(rc_pole)+' '+str(dpsidx)+' '+str(dpsidy)+' '+str(dpsidz)
        # Making so variable class attributes
        self.nbet = nbet
        self.rc = np.ascontiguousarray(rc)
        self.rx = np.ascontiguousarray(rx)
        self.cosx = np.ascontiguousarray(cosx)
        self.cosy = np.ascontiguousarray(cosy)
        self.cosz = np.ascontiguousarray(cosz)
        self.coschi = np.ascontiguousarray(coschi)
        self.area = np.ascontiguousarray(area)
        self.logg = np.ascontiguousarray(logg)
        self.gradx = np.ascontiguousarray(gradx)
        self.grady = np.ascontiguousarray(grady)
        self.gradz = np.ascontiguousarray(gradz)
        logger.log(9, "end")
        return

    def _Velocity_surface(self, phase, velocity=0.):
        """_Velocity_surface(phase, velocity=0.)
        Returns the velocity (in v/c) of each surface element
        of the star.
        
        Positive velocity is away from observer (i.e. redshift).
        Negative velocity is towards observer (i.e. blueshift).

        phase: orbital phase (in orbital fraction; 0: companion 
            in front, 0.5: companion behind).
        velocity: systematic velocity offset to be added (in m/s).
        
        >>> self._Velocity_surface(phase)
        """
        phi = cts.TWOPI*phase
#        # vx = w*y
#        # vy = w*(b-x) # b = semi-major axis
#        # Vx = vx*cos(phi) + vx*sin(phi)
#        # Vy = -vx*sin(phi) + vy*cos(phi)
#        # Rbary: distance between the companion and the barycenter in units of orbital separation
#        Rbary = self.q/(1+self.q)
#        # Kbary: projected velocity amplitude of the companion center of mass in units of c.
#        #    The minus sign is to make the same convention as the findvelocity code: negative velocity means going away from Earth.
#        Kbary = -self.k1 / cts.c
#        Vx = Kbary * (self.rc*self.cosy*np.cos(phi) + (Rbary - self.rc*self.cosx)*np.sin(phi))
#        #print( Vx.min()*cts.c, Vx.max()*cts.c, Vx.mean()*cts.c, (Vx-Vx.mean()).min()*cts.c, (Vx-Vx.mean()).max()*cts.c )
        ## This is the old way
        #Vx = (-self.k1+velocity)/cts.c * ( self.omega*self.rc*(1+self.q)/self.q * (-np.cos(phi)*self.cosy + np.sin(phi)*self.cosx) - np.sin(phi) )
        ## This is the new way
        Vx = (self.k1*np.sin(phi)+velocity)/cts.c - self.omega*self.k1*self.rc*(1+self.q)/self.q/cts.c * (np.sin(phi)*self.cosx + np.cos(phi)*self.cosy)
        return Vx

######################## class Star_base ########################

