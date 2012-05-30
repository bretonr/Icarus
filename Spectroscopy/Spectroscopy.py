# Licensed under a 3-clause BSD style license - see LICENSE

__all__ = ["Spectroscopy"]

from ..Utils.import_modules import *
from .. import Utils
from .. import Core
from .. import Atmosphere


######################## class Spectroscopy ########################
class Spectroscopy:
    """Spectroscopy
    This class allows to fit the flux from the primary star
    of a binary system, assuming it is heated by the secondary
    (which in most cases will be a pulsar).
    
    It is meant to deal with spectroscopic data. A set of spectroscopic
    data (i.e. different orbital phases) is read. For each data set, one can
    calculate the predicted flux of the model at every data point (i.e.
    for a given orbital phase).
    """
    def __init__(self, atmo_fln, data_fln, nalf, porb, x2sini, edot, phase_offset=-0.25, check=True, seeing=-1, oversample=None, smooth=None, savememory=True):
        """__init__(atmo_fln, data_fln, nalf, porb, x2sini, edot, phase_offset=-0.25, check=True, seeing=-1, savememory=True)
        This class allows to fit the flux from the primary star
        of a binary system, assuming it is heated by the secondary
        (which in most cases will be a pulsar).
    
        It is meant to deal with spectroscopic data. A set of spectroscopic
        data (i.e. different orbital phases) is read. For each data set, one can
        calculate the predicted flux of the model at every data point (i.e.
        for a given orbital phase).
        
        atmo_fln: A file containing the grid model information for the whole
            data set. The format of each line of the file is as follows:
                descriptor_name, wavelength cut low, wavelength cut high, grid file
            NOTE: Can also be an atmo_grid instance.
        data_fln: A file containing the information for each data set.
            The format of the file is as follows:
                descriptor_name, orbital phase, column wavelength, column flux,
                column error flux, data file
        nalf: The number of surface slice. Defines how coarse/fine the
            surface grid is.
        porb: Orbital period of the system in seconds.
        x2sini: Projected semi-major axis of the secondary (pulsar)
            in light-second.
        edot: Irradiated energy from the secondary (i.e. spin-down
            luminosity). This is only used for the calculation of the
            irradiation efficiency so it does not enter in the modeling
            itself.
        phase_offset (-0.25): Value to be added to the orbital phase in order
            to have phase 0.0 and 0.5 being conjunction times, with 0.0 the eclipse.
        check (True): Performs a check on the wavelength of the data
            vs. the model so nothing is out of bound.
        seeing (-1): The seeing factor. -1 will use the default value.
        oversample (None): Oversampling factor (integer). If provided, a cubic spline
            interpolation will be performed in order to oversample the grid in the
            wavelength dimension by a factor 'oversample'.
        smooth (None): If provided, the grid will be smoothed with a Gaussian with
            a sigma equals to 'smooth' in the wavelength dimension.
        savememory (False): If true, will keep the mu factors on the
            side and will account for them at the flux calculation time
            in the modified Inter8 function. Allows to save memory in the
            atmosphere grid.
        
        >>> fit = Spectroscopy(atmo_fln, data_fln, nalf, porb, x2sini, edot)
        """
        # We define some class attributes.
        self.porb = porb
        self.x2sini = x2sini
        self.edot = edot
        # We read the data.
        print( 'Reading spectral data' )
        self.__Read_data(data_fln, phase_offset=phase_offset)
        # We read the atmosphere models with the atmo_grid class
        print( 'Reading atmosphere grid' )
        if type(atmo_fln) == type('string'):
            self.__Read_atmo(atmo_fln, oversample=oversample, smooth=smooth, savememory=savememory)
        else:
            self.atmo_grid = atmo_fln
        # We keep in mind the number of datasets
        self.ndataset = len(self.data['phase'])
        # We initialize some important class attributes.
        print( 'Initializing the lightcurve attribute' )
        self.lightcurve = Star.Star(nalf, atmo_grid=self.atmo_grid)
        print( 'Performing some more initialization' )
        self.Initialize(check=check, seeing=seeing)
        print( 'Done. Play and have fun...' )

    def Calc_chi2(self, par, func_par=None, full_output=False, a_n=None, trim=[250,200], velocities=0., verbose=False):
        """Calc_chi2(par, func_par=None, full_output=False, a_n=None, trim=[250,200], velocities=0., verbose=False)
        Returns the chi-square of the fit of the data to the model.
        
        par: Parameter list.
            [0]: Orbital inclination in radians.
            [1]: Corotation factor.
            [2]: Roche-lobe filling.
            [3]: Companion temperature.
            [4]: Gravity darkening coefficient.
            [5]: K (projected velocity semi-amplitude) in m/s.
            [6]: Front side temperature.
            [7]: Systematic velocity offset in m/s.
        func_par (None): Function that takes the parameter vector and
            returns the parameter vector. This allow for possible constraints
            on the parameters. The vector returned by func_par must have a length
            equal to the number of expected parameters.
        full_output (False): If true, the continuum polynomial fit is returned
            as well as the polynomial coefficients.
        a_n (None): If provided, will multiple the model spectra
            by polynomials having coefficients provided in a_n.
        trim ([250,100]): The number of data points to be trimed at the beginning
            and at the end of the spectra. This avoid problems with doppler
            shifting and poor data points.
        velocities (0.): A scalar/vector of velocities to add to the
            pre-computed ones in m/s.
        verbose (False): verbosity.
        
        >>> chi2 = self.Calc_chi2([PIBYTWO,1.,0.9,4000.,0.08,1.4,0.07,10.,0.])
        """
        if func_par is not None:
            par = func_par(par)
        pred_flux = self.Get_flux(par, verbose=verbose, velocities=velocities)
        norm = []
        chi2 = []
        poly_coeff = []
        for i in numpy.arange(self.ndataset):
            if a_n is None:
                poly_coeff.append( Utils.GPolynomial_fit(self.data['flux'][i][trim[0]:-trim[1]], self.data['wavelength'][i][trim[0]:-trim[1]]-self.data['wavelength'][i].mean(), err=self.data['err'][i][trim[0]:-trim[1]], coeff=3, Xfnct=pred_flux[i][trim[0]:-trim[1]], Xfnct_offset=False) )
            else:
                poly_coeff.append( a_n[i] )
            p_n = numpy.poly1d(poly_coeff[i])
            norm.append( p_n(self.data['wavelength'][i]-self.data['wavelength'][i].mean()) )
            chi2.append( ( ( (self.data['flux'][i][trim[0]:-trim[1]] - norm[i][trim[0]:-trim[1]]*pred_flux[i][trim[0]:-trim[1]]) / self.data['err'][i][trim[0]:-trim[1]])**2).sum() )
        
        if full_output is True:
            return chi2, norm, poly_coeff
        else:
            return numpy.array(chi2).sum()

    def Calc_chi2_doppler(self, par, func_par=None, full_output=False, a_n=None, v_approx=None, trim=[250,200], verbose=False, pred_flux=None):
        """Calc_chi2_doppler(par, func_par=None, full_output=False, a_n=None, v_approx=None, trim=[250,200], verbose=False, pred_flux=None)
        Returns the chi-square of the fit of the data to the model.
        Will attempt to fit a doppler shift to the whole data set.
        
        par: Parameter list.
            [0]: Orbital inclination in radians.
            [1]: Corotation factor.
            [2]: Roche-lobe filling.
            [3]: Companion temperature.
            [4]: Gravity darkening coefficient.
            [5]: K (projected velocity semi-amplitude) in m/s.
            [6]: Front side temperature.
            [7]: Systematic velocity offset in m/s.
        func_par (None): Function that takes the parameter vector and
            returns the parameter vector. This allow for possible constraints
            on the parameters. The vector returned by func_par must have a length
            equal to the number of expected parameters.
        full_output (False): If true, the continuum polynomial fit is returned
            as well as the polynomial coefficients.
        a_n (None): If provided, will multiple the model spectra
            by polynomials having coefficients provided in a_n.
        v_approx (None): An approximate velocity for the best velocity
            search algorithm in m/s. Should be the binary system velocity.
            (Must be a float)
        trim ([250,100]): The number of data points to be trimed at the beginning
            and at the end of the spectra. This avoid problems with doppler
            shifting and poor data points.
        verbose (False): Verbosity.
        pred_flux (None): A list of predicted flux values for each data set.
            If provided, no computation of the flux will be made.
        
        >>> chi2 = self.Calc_chi2_doppler([PIBYTWO,1.,0.9,4000.,0.08,1.4,0.07,10.,0.])
        """
        if func_par is not None:
            par = func_par(par)
        if pred_flux is None:
           pred_flux = numpy.array(self.Get_flux(par, rebin=False, verbose=verbose))

        norm = [0]*self.ndataset
        chi2 = numpy.empty(self.ndataset)
        poly_coeff = [0]*self.ndataset
        if v_approx is None:
            v_approx = 0.
        def get_chi2(v):
            flux = self.Rebin(self.Doppler_shift(pred_flux.T.copy(), v[0]).T)
            for i in numpy.arange(self.ndataset):
                if a_n is None:
                    poly_coeff[i] = Utils.GPolynomial_fit(self.data['flux'][i][trim[0]:-trim[1]], self.data['wavelength'][i][trim[0]:-trim[1]]-self.data['wavelength'][i].mean(), err=self.data['err'][i][trim[0]:-trim[1]], coeff=3, Xfnct=flux[i][trim[0]:-trim[1]], Xfnct_offset=False)
                else:
                    poly_coeff[i] = a_n[i]
                p_n = numpy.poly1d(poly_coeff[i])
                norm[i] = p_n(self.data['wavelength'][i]-self.data['wavelength'][i].mean())
                chi2[i] = ( ( (self.data['flux'][i][trim[0]:-trim[1]] - norm[i][trim[0]:-trim[1]]*flux[i][trim[0]:-trim[1]]) / self.data['err'][i][trim[0]:-trim[1]])**2).sum()
            return chi2.sum()
        best_vshift = scipy.optimize.fmin(get_chi2, v_approx, full_output=0, maxiter=100, disp=0)
        
        if verbose:
            print( "best_vshift (km/s): " + str(best_vshift/1000) + ", chi2: " + str(chi2.sum()) )
        
        if full_output is True:
            return chi2, norm, poly_coeff, best_vshift
        else:
            return chi2.sum()

    def Calc_chi2_doppler_inds(self, par, func_par=None, full_output=False, a_n=None, v_approx=None, trim=[250,200], verbose=False):
        """Calc_chi2_doppler_inds(par, func_par=None, full_output=False, a_n=None, v_approx=None, trim=[250,200], verbose=False)
        Returns the chi-square of the fit of the data to the model.
        Will attempt to fit a doppler shift to each data.
        
        par: Parameter list.
            [0]: Orbital inclination in radians.
            [1]: Corotation factor.
            [2]: Roche-lobe filling.
            [3]: Companion temperature.
            [4]: Gravity darkening coefficient.
            [5]: K (projected velocity semi-amplitude) in m/s.
            [6]: Front side temperature.
            [7]: Systematic velocity offset in m/s.
        func_par (None): Function that takes the parameter vector and
            returns the parameter vector. This allow for possible constraints
            on the parameters. The vector returned by func_par must have a length
            equal to the number of expected parameters.
        full_output (False): If true, the continuum polynomial fit is returned
            as well as the polynomial coefficients.
        a_n (None): If provided, will multiple the model spectra
            by polynomials having coefficients provided in a_n.
        v_approx (None): An approximate velocity for the best velocity
            search algorithm in m/s. Should be the binary system velocity.
            (Must be a vector whose length equal the number of data)
        trim ([250,100]): The number of data points to be trimed at the beginning
            and at the end of the spectra. This avoid problems with doppler
            shifting and poor data points.
        verbose (False): Verbosity.
        
        >>> chi2 = self.Calc_chi2_doppler_inds([PIBYTWO,1.,0.9,4000.,0.08,1.4,0.07,10.,0.])
        """
        if func_par is not None:
            par = func_par(par)
        pred_flux = numpy.array(self.Get_flux(par, rebin=False, verbose=verbose))

        norm = [0]*self.ndataset
        chi2 = numpy.empty(self.ndataset)
        err = numpy.empty(self.ndataset)
        poly_coeff = [0]*self.ndataset
        best_vshift = numpy.empty(self.ndataset)
        if v_approx is None:
            v_approx = [0.]*self.ndataset
        for i in numpy.arange(self.ndataset):
#            print( "Fitting velocity for i: "+str(i) )
            def get_chi2(v):
                flux = self.Rebin(self.Doppler_shift(pred_flux[i].copy(), v[0]), ind=i)
                if a_n is None:
                    poly_coeff[i] = Utils.GPolynomial_fit(self.data['flux'][i][trim[0]:-trim[1]], self.data['wavelength'][i][trim[0]:-trim[1]]-self.data['wavelength'][i].mean(), err=self.data['err'][i][trim[0]:-trim[1]], coeff=3, Xfnct=flux[trim[0]:-trim[1]], Xfnct_offset=False)
                else:
                    poly_coeff[i] = a_n[i]
                p_n = numpy.poly1d(poly_coeff[i])
                norm[i] = p_n(self.data['wavelength'][i]-self.data['wavelength'][i].mean())
                #chi2[i] = ( ( (self.data['flux'][i][trim[0]:-trim[1]] - norm[i][trim[0]:-trim[1]]*flux[trim[0]:-trim[1]]) / self.data['err'][i][trim[0]:-trim[1]])**2).sum()
                #return chi2[i]
                return (self.data['flux'][i][trim[0]:-trim[1]] - norm[i][trim[0]:-trim[1]]*flux[trim[0]:-trim[1]]) / self.data['err'][i][trim[0]:-trim[1]]
            #vels = numpy.linspace(-500e3,500e3,201)
            #vchi2 = numpy.r_[[(get_chi2([v])**2).sum() for v in vels]]
            #chi2[i], best_vshift[i], err[i] = icarus.Err_velocity(vchi2, vels, self.data['flux'][i].size-trim[0]-trim[1], clip=40e3, redchi2_unity=True, verbose=False)
            #best_vshift[i], fopt, niter, funcalls, warnflag = scipy.optimize.fmin(get_chi2, [v_approx[i]], full_output=1, maxiter=100, disp=0)
            best_vshift[i], cov_x, infodict, mesg, ier = scipy.optimize.leastsq(get_chi2, [v_approx[i]], full_output=1)
            err[i] = numpy.sqrt(cov_x[0][0])
            chi2[i] = (infodict['fvec']**2).sum()
#            print( best_vshift[i], fopt, niter, funcalls, warnflag )
        
        if verbose:
#            print( 'ind chi2: ', chi2 )
            print( 'chi2: '+str(chi2.sum()) )
#            print( 'err: ', err )
        if full_output is True:
            return chi2, norm, poly_coeff, best_vshift, err
        else:
            return chi2.sum()

    def Calc_chi2_doppler_offset(self, par, func_par=None, full_output=False, a_n=None, v_approx=None, trim=[250,200], verbose=False):
        """Calc_chi2_doppler_offset(par, func_par=None, full_output=False, a_n=None, v_approx=None, trim=[250,200], verbose=False)
        Returns the chi-square of the fit of the data to the model.
        Will attempt to fit a doppler shift to each data and will
        calculate a global offset, which adds as a chi2 penalty to
        the total chi2.
        
        par: Parameter list.
            [0]: Orbital inclination in radians.
            [1]: Corotation factor.
            [2]: Roche-lobe filling.
            [3]: Companion temperature.
            [4]: Gravity darkening coefficient.
            [5]: K (projected velocity semi-amplitude) in m/s.
            [6]: Front side temperature.
            [7]: Systematic velocity offset in m/s.
        func_par (None): Function that takes the parameter vector and
            returns the parameter vector. This allow for possible constraints
            on the parameters. The vector returned by func_par must have a length
            equal to the number of expected parameters.
        full_output (False): If true, the continuum polynomial fit is returned
            as well as the polynomial coefficients.
        a_n (None): If provided, will multiple the model spectra
            by polynomials having coefficients provided in a_n.
        v_approx (None): An approximate velocity for the best velocity
            search algorithm in m/s. Should be the binary system velocity.
            (Must be a float)
        trim ([250,100]): The number of data points to be trimed at the beginning
            and at the end of the spectra. This avoid problems with doppler
            shifting and poor data points.
        verbose (False): Verbosity.
        
        >>> chi2 = self.Calc_chi2_doppler_offset([PIBYTWO,1.,0.9,4000.,0.08,1.4,0.07,10.,0.])
        """
        if func_par is not None:
            par = func_par(par)
        pred_flux = numpy.array(self.Get_flux(par, rebin=False, verbose=verbose))

        norm = [0]*self.ndataset
        chi2 = numpy.empty(self.ndataset)
        poly_coeff = [0]*self.ndataset
        best_vshift = numpy.empty(self.ndataset)
        err = numpy.empty(self.ndataset)
        if v_approx is None:
            v_approx = 0.
        # Here we iterate over the data set to fit each spectrum for the best-fit velocity
        for i in numpy.arange(self.ndataset):
#            print( "Fitting velocity for i: "+str(i) )
            def get_chi2(v):
                # Actually, here we return the residuals, not the chi2
                flux = self.Rebin(self.Doppler_shift(pred_flux[i].copy(), v[0]), ind=i)
                if a_n is None:
                    poly_coeff[i] = icarus.GPolynomial_fit(self.data['flux'][i][trim[0]:-trim[1]], self.data['wavelength'][i][trim[0]:-trim[1]]-self.data['wavelength'][i].mean(), err=self.data['err'][i][trim[0]:-trim[1]], coeff=3, Xfnct=flux[trim[0]:-trim[1]], Xfnct_offset=False)
                else:
                    poly_coeff[i] = a_n[i]
                p_n = numpy.poly1d(poly_coeff[i])
                norm[i] = p_n(self.data['wavelength'][i]-self.data['wavelength'][i].mean())
                return (self.data['flux'][i][trim[0]:-trim[1]] - norm[i][trim[0]:-trim[1]]*flux[trim[0]:-trim[1]]) / self.data['err'][i][trim[0]:-trim[1]]
            best_vshift[i], cov_x, infodict, mesg, ier = scipy.optimize.leastsq(get_chi2, v_approx, full_output=1)
            ##print best_vshift[i], cov_x, ier
            err[i] = numpy.sqrt(cov_x[0][0])
            chi2[i] = (infodict['fvec']**2).sum()
        # Here we find the best-fit systematic velocity resulting form the individual fits
        # The more closely the individual fits follow the predicted orbital velocities, the better the fit.
        # The orbital contribution is already included in the model spectrum. One therefore expects all the
        # individual fits to have the same systematic velocity.
        err = numpy.sqrt(err**2 + self.data['v_offset_err']**2)
        #err = err
        ((sys_vel,tmp), chi2_offset, rank, s) = bretonr_utils.fit_linear(best_vshift, err=err, m=0.)
        ##print chi2, chi2_offset, chi2.sum()+chi2_offset

        if verbose:
            print( "Velocities (km/s): ", best_vshift/1000, " System velocity (km/s): ", sys_vel/1000 )
            print( "chi2: ", chi2, " chi2_offset: ", chi2_offset, " chi2 tot: ", chi2.sum()+chi2_offset )
        
        if full_output:
            return chi2.sum()+chi2_offset, norm, poly_coeff, best_vshift, err, sys_vel, chi2, chi2_offset
        else:
            return chi2.sum()+chi2_offset

    def Doppler_shift(self, flux, v):
        """Doppler_shift(flux, v)
        Shifts the spectrum according to the velocity v in m/s.
        """
        # Here we divide the velocity by the speed of light because this is the units of atmo_grid
        bin = (v/cts.c)/self.atmo_grid.v
        ws = bin%1
        bin = numpy.floor(bin)
        if bin >= 0:
            flux[:flux.shape[0]-bin-1] = flux[bin:-1]*(1-ws) + flux[bin+1:]*ws
        else:
            bin *= -1
            flux[bin:-1] = flux[:flux.shape[0]-bin-1]*(1-ws) + flux[1:flux.shape[0]-bin]*ws
        return flux

    def Get_flux(self, par, func_par=None, rebin=True, velocities=0., verbose=False):
        """Get_flux(par, func_par=None, rebin=True, velocities=0., verbose=False)
        Returns the predicted flux by the model evaluated at the
        observed values in the data set.
        
        par: Parameter list.
            [0]: Orbital inclination in radians.
            [1]: Corotation factor.
            [2]: Roche-lobe filling.
            [3]: Companion temperature.
            [4]: Gravity darkening coefficient.
            [5]: K (projected velocity semi-amplitude) in m/s.
            [6]: Front side temperature.
            [7]: Systematic velocity offset in m/s.
        func_par (None): Function that takes the parameter vector and
            returns the parameter vector. This allow for possible constraints
            on the parameters. The vector returned by func_par must have a length
            equal to the number of expected parameters.
        rebin (True): If true, will apply the Rebin method in order to
            have the model spectrum match the observed wavelengths.
        velocities (0.): A scalar/vector of velocities to add to the
            pre-computed ones (in m/s unit).
        verbose (False): If true will display the list of parameters.
        
        Note: tirr = (par[6]**4 - par[3]**4)**0.25
        
        >>> self.Get_flux([PIBYTWO,1.,0.9,4000.,0.08,1.4,0.07,0.])
        """
        if func_par is not None:
            par = func_par(par)
        # Making sure the values are not out-of-bound
        if par[0] < 0.:
            par[0] = 0.
        elif par[0] > PIBYTWO:
            par[0] = PIBYTWO
        if par[2] > 1.: par[2] = 1.
        if par[3] < numpy.exp(self.atmo_grid.grid_teff[0]):
            par[3] = numpy.exp(self.atmo_grid.grid_teff[0])
        elif par[3] > numpy.exp(self.atmo_grid.grid_teff[-1]):
            par[3] = numpy.exp(self.atmo_grid.grid_teff[-1])
        if par[5] < 0.: par[5] = 0.
        if par[6] < par[3]: par[6] = par[3]
        if par[6] < numpy.exp(self.atmo_grid.grid_teff[0]):
            par[6] = numpy.exp(self.atmo_grid.grid_teff[0])
        elif par[6] > numpy.exp(self.atmo_grid.grid_teff[-1]):
            par[6] = numpy.exp(self.atmo_grid.grid_teff[-1])
        # The velocity offset is:
        #  the 'v_offset' from the data, which is the barycentric correction determined from the optical companion
        #  the 'par[7]' is from the fit parameters, though usually set to 0.
        #  the 'velocities' are extra velocity contributions.
        v_offset = self.data['v_offset'] + par[7] + velocities
        
        q = par[5] * self.K_to_q
        tirr = (par[6]**4 - par[3]**4)**0.25
#        q = Mass_ratio(self.mass_function, par[5], par[0])
#        lirr = Lirr(par[6], self.edot, self.x2sini, q, par[0])
        if verbose:
            print( "#####\n" + str(par[0]) + ", " + str(par[1]) + ", " + str(par[2]) + ", " + str(par[3]) + ", " + str(par[4]) + ", " + str(par[5]) + ", " + str(par[6]) + ", " + str(par[7]) + "\n" + "q: " + str(q) + ", tirr: " + str(tirr)  )
        self.lightcurve.Make_surface(q=q, omega=par[1], filling=par[2], temp=par[3], tempgrav=par[4], tirr=tirr, porb=self.porb, k1=par[5], incl=par[0])
        flux = [self.lightcurve.Flux_doppler(self.data['phase'][i], velocity=v_offset[i]) for i in numpy.arange(self.ndataset)]
        if rebin is True:
            flux = self.Rebin(flux)
        return flux

    def Get_flux_ind(self, par, orb_phs=0., ind=0, rebin=True, verbose=False):
        """Get_flux_ind(par, rebin=True, verbose=False)
        Returns the predicted flux by the model evaluated at the
        observed values in the data set.
        
        par: Parameter list.
            [0]: Orbital inclination in radians.
            [1]: Corotation factor.
            [2]: Roche-lobe filling.
            [3]: Companion temperature.
            [4]: Gravity darkening coefficient.
            [5]: K (projected velocity semi-amplitude) in m/s.
            [6]: Front side temperature.
            [7]: Systematic velocity offset in m/s.
        orb_phs (0.): Orbital phase (0 -> eclipse)
        ind (0): Index of the data to use (for the wavelengths).
        rebin (True): If true, will apply the Rebin method in order to
            have the model spectrum match the observed wavelengths.
        verbose (False): If true will display the list of parameters.
        
        Note: tirr = (par[6]**4 - par[3]**4)**0.25
        
        >>> self.Get_flux_ind([PIBYTWO,1.,0.9,4000.,0.08,1.4,0.07,0.])
        """
        q = par[5] * self.K_to_q
        tirr = (par[6]**4 - par[3]**4)**0.25
        if verbose:
            print( "#####\n" + str(par[0]) + ", " + str(par[1]) + ", " + str(par[2]) + ", " + str(par[3]) + ", " + str(par[4]) + ", " + str(par[5]) + ", " + str(par[6]) + ", " + str(par[7]) + "\n" + "q: " + str(q) + ", tirr: " + str(tirr)  )
        self.lightcurve.Make_surface(q=q, omega=par[1], filling=par[2], temp=par[3], tempgrav=par[4], tirr=tirr, porb=self.porb, k1=par[5], incl=par[0])
        flux = self.lightcurve.Flux_doppler(par[0], orb_phs, velocity=par[7])
        if rebin is True:
            flux = self.Rebin(flux, ind=ind)
        return flux

    def Initialize(self, check=True, seeing=-1):
        """Initialize(check=True, seeing=-1)
        Initializes and stores some important variables
        
        check (True): Performs a check on the wavelength of the data
            vs. the model so nothing is out of bound.
        seeing (-1): The seeing factor. -1 will use the default value.
        """
        # We calculate the constant for the conversion of K to q (observed
        # velocity semi-amplitude to mass ratio, with K in m/s)
        self.K_to_q = icarus.Get_K_to_q(self.porb, self.x2sini)
        # Here we pre-calculate the wavelengths, interpolation indices
        # and weights for the rebining of the log-spaced atmo_grid data
        # to linear
        self.wavelength = []
        self.binfactor = []
        self.ws_loglin = []
        self.inds_loglin = []
        self.ws_rebin = []
        self.inds_rebin = []
        self.sigma = []
        for i in numpy.arange(self.ndataset):
            if (check is True) and (self.data['wavelength'][i][0] < self.atmo_grid.grid_lam[0]):
                inds = self.data['wavelength'][i] > self.atmo_grid.grid_lam[0]
                self.data['wavelength'][i] = self.data['wavelength'][i][inds]
                self.data['flux'][i] = self.data['flux'][i][inds]
                self.data['err'][i] = self.data['err'][i][inds]
            stepsize = self.data['wavelength'][i][1]-self.data['wavelength'][i][0]
            # binfactor is the oversampling factor of the model vs observed spectrum
            tmp = numpy.floor(stepsize/(self.atmo_grid.grid_lam[1]-self.atmo_grid.grid_lam[0]))
            if tmp < 1: tmp = 1
            self.binfactor.append( tmp )
            stepsize /= self.binfactor[i]
            self.wavelength.append(  numpy.arange(self.data['wavelength'][i][0],self.data['wavelength'][i][-1]+stepsize/2,stepsize) )
            # To rebin from log to lin
            ws, inds = icarus.Getaxispos_vector(self.atmo_grid.grid_lam, self.wavelength[i]+0.0)
            self.ws_loglin.append(ws)
            self.inds_loglin.append(inds)
            # To rebin to the observed wavelengths
            ws, inds = icarus.Getaxispos_vector(self.wavelength[i], self.data['wavelength'][i]+0.0)
            self.ws_rebin.append(ws)
            self.inds_rebin.append(inds)
            
            # sigma is for the gaussian convolution to blur the model spectrum
            # to the observed seeing
            if seeing == -1:
                self.sigma.append( 2.2*0.625/stepsize )
            else:
                self.sigma.append( seeing )
        return

    def Plot(self, par, ind, device='/XWIN', trim=[250,200], velocities=0., verbose=True):
        """Plot(par, ind, device='/XWIN', trim=[250,200], velocities=0., verbose=True)
        Plots the observed and predicted values along with the
        light curve.
        par: Parameter list.
            [0]: Orbital inclination in radians.
            [1]: Corotation factor.
            [2]: Roche-lobe filling.
            [3]: Companion temperature.
            [4]: Gravity darkening coefficient.
            [5]: K (projected velocity semi-amplitude) in m/s.
            [6]: Front side temperature.
            [7]: Systematic velocity offset in m/s.
        ind: Index of the data to be plotted
        device ('/XWIN'): Device driver for Pgplot (can be '/XWIN',
            'filename.ps/PS', 'filename.ps./CPS', '/AQT' (on mac only)).
        trim ([250,100]): The number of data points to be trimed at the beginning
            and at the end of the spectra. This avoid problems with doppler
            shifting and poor data points.
        velocities (True): Velocities to be added to the spectra in m/s.
        
        >>> self.Plot([PIBYTWO,1.,0.9,4000.,0.08,1.4,0.07,10.,0.])
        """
        if verbose:
            print( "##### ##### Plot data " + str(ind) + ", orb. phase: " + str(self.data['phase'][ind]) )
        chi2, norm, poly_coeff = self.Calc_chi2(par, full_output=True, trim=trim, velocities=velocities, verbose=verbose)
        par_new = numpy.asarray(par)
        ######### should this come before getting the chi2 and poly_coeff?
        if type(velocities) == type(4.):
            par_new[7] += self.data['v_offset'][ind] + velocities
        else:
            par_new[7] += self.data['v_offset'][ind] + velocities[ind]
        pred_flux = self.Get_flux_ind(par_new, orb_phs=self.data['phase'][ind], ind=ind)
        a = self.data['flux'][ind][trim[0]:-trim[1]]
        b = pred_flux[trim[0]:-trim[1]]*norm[ind][trim[0]:-trim[1]]
        c = a-b
        plotxy(a, self.data['wavelength'][ind][trim[0]:-trim[1]], color=1, rangey=[c.min(), a.max()], device=device)
        plotxy(b, self.data['wavelength'][ind][trim[0]:-trim[1]], color=2)
        plotxy(c, self.data['wavelength'][ind][trim[0]:-trim[1]], color=4)
        return

    def Pretty_print(self, par, make_surface=True, verbose=True):
        """
        Return a nice representation of the important
        parameters.
        
        par: Parameter list.
            [0]: Orbital inclination in radians.
            [1]: Corotation factor.
            [2]: Roche-lobe filling.
            [3]: Companion temperature.
            [4]: Gravity darkening coefficient.
            [5]: K (projected velocity semi-amplitude) in m/s.
            [6]: Front side temperature.
            [7]: Systematic velocity offset in m/s.
        make_surface (True): Whether to recalculate the 
            surface of the star or not.
        verbose (True): Output the nice representation
            of the important parameters or just return them
            as a list.
        """
        incl = par[0]
        corot = par[1]
        fill = par[2]
        temp_back = par[3]
        gdark = par[4]
        K = par[5]
        temp_front = par[6]
        vel_sys = par[7]
        q = K * self.K_to_q
        tirr = (temp_front**4 - temp_back**4)**0.25
        if make_surface:
            self.lightcurve.Make_surface(q=q, omega=corot, filling=fill, temp=temp_back, tempgrav=gdark, tirr=tirr, porb=self.porb, k1=K, incl=incl)
        separation = self.lightcurve.separation
        roche = self.lightcurve.Roche()
        Mwd = self.lightcurve.mass1
        Mns = self.lightcurve.mass2
        # below we transform sigma from W m^-2 K^-4 to erg s^-1 cm^-2 K^-4
        # below we transform the separation from m to cm
        Lirr = tirr**4 * (cts.sigma*1e3) * (separation*100)**2 * 4*PI
        eff = Lirr/self.edot
        # we convert Lirr in Lsun units
        Lirr /= 3.839e33
        if verbose:
            print( "##### Pretty Print #####" )
            print( "%9.7f, %3.1f, %9.7f, %10.5f, %4.2f, %9.7f, %9.7f, %12.10f" %tuple(par) )
            print( "" )
            print( "Corotation factor: %4.2f" %corot )
            print( "Gravity Darkening: %5.3f" %gdark )
            print( "" )
            print( "Filling factor: %6.4f" %fill )
            print( "Orbital separation: %5.4e km" %(separation/1000) )
            print( "Roche lobe size: %6.4f (orb. sep.)" %roche )
            print( "" )
            print( "Irradiation efficiency: %6.4f" %eff )
            print( "Irration luminosity: %5.4e Lsun" %Lirr )
            print( "Backside temperature: %7.2f K" %temp_back )
            print( "Frontside temperature: %7.2f (tabul.), %7.2f (approx.) K" %(numpy.exp(self.lightcurve.logteff.max()),temp_front) )
            print( "" )
            print( "Inclination: %5.3f rad (%6.2f deg)" %(incl,incl*RADTODEG) )
            print( "System velocity: %7.3f km/s" %(vel_sys/1000) )
            print( "K: %7.3f km/s" %(K/1000) )
            print( "" )
            print( "Mass ratio: %6.3f" %q )
            print( "Mass NS: %5.3f Msun" %Mns )
            print( "Mass Comp: %5.3f Msun" %Mwd )
        return numpy.r_[corot,gdark,fill,separation,roche,eff,tirr,temp_back,numpy.exp(self.lightcurve.logteff.max()),temp_front,incl,incl*RADTODEG,vel_sys,K,q,Mns,Mwd]

    def __Read_atmo(self, atmo_fln, oversample=None, smooth=None, savememory=True):
        """__Read_atmo(atmo_fln, oversample=None, smooth=None, savememory=True)
        Reads the atmosphere model data.
        
        atmo_fln: A file containing the grid model information for the whole
            data set. The format of each line of the file is as follows:
            {descriptor_name, wavelength cut low, wavelength cut high,
            grid file}. "wavelength cut low/high" are in the same units
            as the grid (probably angstrom). "grid file" can be a wildcard.
        oversample (None): Oversampling factor (integer). If provided, a cubic spline
            interpolation will be performed in order to oversample the grid in the
            wavelength dimension by a factor 'oversample'.
        smooth (None): If provided, the grid will be smoothed with a Gaussian with
            a sigma equals to 'smooth' in the wavelength dimension.
        savememory (False): If true, will keep the mu factors on the
            side and will account for them at the flux calculation time
            in the modified Inter8 function. Allows to save memory in the
            atmosphere grid.
        
        >>> self.__Read_atmo(atmo_fln)
        """
        f = open(atmo_fln,'r')
        lines = f.readlines()
        tmp = lines[0].split()
        flns = glob.glob(tmp[3])
        self.atmo_grid = Atmosphere.Atmo_grid_lithium_doppler(flns, oversample=oversample, smooth=smooth, wave_cut=[float(tmp[1]),float(tmp[2])], savememory=savememory)
        return

    def __Read_data(self, data_fln, phase_offset=-0.25):
        """__Read_data(self, data_fln, phase_offset=-0.25)
        Reads the photometric data.
        
        data_fln: A file containing the information for each data set.
            The format of the file is as follows:
                descriptor name
                orbital phase
                column name wavelength
                column name flux
                column name error flux
                data file
                barycenter offset (i.e. measured velocity for optical 
                    companion in km/s)
                barycenter offset error (i.e. measured velocity error 
                    for optical companion in km/s)
                approximate velocity (i.e. measure velocity for pulsar companion
                    with velocity_find in km/s)
        phase_offset (-0.25): Value to be added to the orbital phase in order
            to have phase 0.0 and 0.5 being conjunction times, with 0.0 the eclipse.
        
        >>> self.__Read_data(data_fln)
        """
        import pyfits
        f = open(data_fln,'r')
        lines = f.readlines()
        self.data = {'wavelength':[], 'flux':[], 'phase':[], 'err':[], 'v_offset':[], 'v_offset_err':[], 'a_n':[], 'v_approx':[], 'fln':[]}
        for line in lines:
            if not line.startswith('#'):
                tmp = line.split()
                # We try to read the data as a fits table, if crash assumes ascii file
                try:
                    hdulist = pyfits.open(tmp[5])
                    tbdata = hdulist[1].data.field(tmp[2])
                    inds = numpy.isfinite(tbdata).nonzero()[0]
                    if tbdata[0] > tbdata[1]:
                        inds = inds[::-1]
                    self.data['wavelength'].append(numpy.float64(tbdata[inds]))
                    self.data['flux'].append(numpy.float64(hdulist[1].data.field(tmp[3])[inds]))
                    self.data['err'].append(numpy.float64(hdulist[1].data.field(tmp[4])[inds]))
    #                self.data['wavelength'].append(tbdata[inds])
    #                self.data['flux'].append(hdulist[1].data.field(tmp[3])[inds])
    #                self.data['err'].append(hdulist[1].data.field(tmp[4])[inds])
                except:
                    d = numpy.loadtxt(tmp[5], usecols=[int(tmp[2]),int(tmp[3]),int(tmp[4])], unpack=True)
                    inds = numpy.isfinite(d[0]).nonzero()[0]
                    if d[0,0] > d[0,1]:
                        inds = inds[::-1]
                    self.data['wavelength'].append(d[0,inds])
                    self.data['flux'].append(d[1,inds])
                    self.data['err'].append(d[2,inds])
                self.data['fln'].append(tmp[5])
                self.data['phase'].append((float(tmp[1])+phase_offset)%1)
                # We have to convert the velocities from km/s to m/s
                self.data['v_offset'].append(float(tmp[6]) * 1000)
                self.data['v_offset_err'].append(float(tmp[7]) * 1000)
                self.data['v_approx'].append(float(tmp[8]) * 1000)
                self.data['a_n'].append(numpy.array(tmp[9:], dtype=float))
        self.data['phase'] = numpy.array(self.data['phase'])
        self.data['v_offset'] = numpy.array(self.data['v_offset'])
        self.data['v_offset_err'] = numpy.array(self.data['v_offset_err'])
        self.data['v_approx'] = numpy.array(self.data['v_approx'])
        # the wavelenghts are not matching from one observation
        # to another so we can't make arrays
#        self.data['wavelength'] = numpy.array(self.data['wavelength'])
#        self.data['flux'] = numpy.array(self.data['flux'])
#        self.data['err'] = numpy.array(self.data['err'])
        return

    def Rebin(self, flux, ind=None):
        """Rebin(flux, ind=None)
        Takes the atmosphere grid flux, rebin to be linearly space
        in wavelength, gaussian filter to account for seeing, uniform
        filter to average to observed bin size.
        
        sigma: sigma of the gaussian filter.
        ind: data set index of the flux to be smoothed.
        """
        if ind is None:
            nflux = [0]*self.ndataset
            for i in numpy.arange(self.ndataset):
                nflux[i] = flux[i].take(self.inds_loglin[i], axis=-1)*(1-self.ws_loglin[i]) + flux[i].take(self.inds_loglin[i]+1, axis=-1)*self.ws_loglin[i]
                nflux[i] = scipy.ndimage.gaussian_filter1d(nflux[i], self.sigma[i])
                nflux[i] = scipy.ndimage.uniform_filter1d(nflux[i], int(self.binfactor[i]))
                nflux[i] = nflux[i].take(self.inds_rebin[i], axis=-1)*(1-self.ws_rebin[i]) + nflux[i].take(self.inds_rebin[i]+1, axis=-1)*self.ws_rebin[i]
        else:
            nflux = flux.take(self.inds_loglin[ind], axis=-1)*(1-self.ws_loglin[ind]) + flux.take(self.inds_loglin[ind]+1, axis=-1)*self.ws_loglin[ind]
            nflux = scipy.ndimage.gaussian_filter1d(nflux, self.sigma[ind])
            nflux = scipy.ndimage.uniform_filter1d(nflux, int(self.binfactor[ind]))
            nflux = nflux.take(self.inds_rebin[ind], axis=-1)*(1-self.ws_rebin[ind]) + nflux.take(self.inds_rebin[ind]+1, axis=-1)*self.ws_rebin[ind]
        return nflux

    def Save_flux(self, fln, par, velocities=0., verbose=False):
        """Save_flux(fln, par, velocities=0., verbose=False)
        Saves the flux in files.
        The format is three columns (wavelength, flux, err).
        Note that the errors are: err = err/flux_obs*flux_pred so
        that the errors are scaled from the observed to modeled flux.
        
        fln: filename, files will be fln.n, where n is the data id.
        par: parameters (see Get_flux for more info).
        velocities (optional): A scalar/vector of velocities to add to the
                    pre-computed ones (in m/s unit).
        verbose: verbosity flag.
        """
        flux = self.Get_flux(par, rebin=True, velocities=velocities, verbose=verbose)
        for i in numpy.arange(self.ndataset):
            err = numpy.abs(self.data['err'][i]/self.data['flux'][i]*flux[i])
            numpy.savetxt(fln+".%02i" %i, numpy.c_[self.data['wavelength'][i],flux[i],err])
        return

######################## class Spectroscopy ########################

