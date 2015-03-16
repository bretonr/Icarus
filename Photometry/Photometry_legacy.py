# Licensed under a 3-clause BSD style license - see LICENSE

__all__ = ["Photometry_legacy"]

from ..Utils.import_modules import *
from .. import Utils
from .. import Core
from .. import Atmosphere


######################## class Photometry ########################
class Photometry_legacy(object):
    """Photometry_legacy
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
        ndiv (int): The number of surface slice. Defines how coarse/fine the
            surface grid is.
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
        
        >>> fit = Photometry(atmo_fln, data_fln, ndiv, porb, x2sini)
        """
        DeprecationWarning("This is the old Photometry class. Use the one from the Photometry instead.")

        # We define some class attributes.
        self.porb = porb
        self.x2sini = x2sini
        self.edot = edot
        # We read the data.
        self._Read_data(data_fln)
        # We read the atmosphere models with the atmo_grid class
        self._Read_atmo(atmo_fln)
        # We make sure that the length of data and atmo_dict are the same
        if len(self.atmo_grid) != len(self.data['id']):
            print 'The number of atmosphere grids and data sets (i.e. photometric bands) do not match!!!'
            return
        else:
            # We keep in mind the number of datasets
            self.ndataset = len(self.atmo_grid)
        # We initialize some important class attributes.
        self._Init_lightcurve(ndiv, read=read)
        self._Setup()

    def Calc_chi2(self, par, offset_free=1, func_par=None, nsamples=None, influx=False, full_output=False, verbose=False):
        """Calc_chi2(par, offset_free=1, func_par=None, nsamples=None, influx=False, full_output=False, verbose=False)
        Returns the chi-square of the fit of the data to the model.
        
        par (list/array): Parameter list.
            [0]: Orbital inclination in radians.
            [1]: Corotation factor.
            [2]: Roche-lobe filling.
            [3]: Companion temperature.
            [4]: Gravity darkening coefficient.
            [5]: K (projected velocity semi-amplitude) in m/s.
            [6]: Front side temperature or irradiation temperature.
                The irradiation temperature is in the case of the
                photometry_modeling_temperature class.
            [7]: Distance modulus (can be None).
            [8]: Absorption A_V (can be None).
            Note: DM and A_V can be set to None. In which case, if
            offset_free = 1, these parameters will be fit for.
            Note: Can also be a dictionary:
                par.keys() = ['av','corotation','dm','filling','gravdark','incl','k1','tday','tnight']
        offset_free (int):
            1) offset_free = 0:
                If the offset is not free and the DM and A_V are specified, the chi2
                is calculated directly without allowing an offset between the data and
                the bands.
                The full chi2 should be:
                    chi2 = sum[ w_i*(off_i-dm-av*C_i)**2]
                        + w_dm*(dm-dm_obs)**2 
                        + w_av*(av-av_obs)**2,     with w = 1/sigma**2
                The extra terms (i.e. dm-dm_obs and av-av_obs) should be included
                as priors.
            1) offset_free = 1:
                The model light curves are fitted to the data with an arbitrary offset
                for each band. After, a post-fit is performed in order to adjust the offsets
                of the curves accounting for the fact that the absolute calibration of the
                photometry may vary.
                Note:
                The errors should be err**2 = calib_err**2 + 1/sum(flux_err)**2
                but we neglect the second term because it is negligeable.
        func_par (None): Function that takes the parameter vector and
            returns the parameter vector. This allow for possible constraints
            on the parameters. The vector returned by func_par must have a length
            equal to the number of expected parameters.
        nsamples (None): Number of points for the lightcurve sampling.
            If None, the lightcurve will be sampled at the observed data
            points.
        influx (False): If true, will calculate the fit between the data and the
            model in the flux domain.
        full_output (bool): If true, will output a dictionnary of additional parameters.
            'offset' (array): the calculated offset for each band.
            'par' (array): the input parameters (useful if one wants to get the optimized
                values of DM and A_V.
            'res' (array): the fit residuals.
        verbose (bool): If true will display the list of parameters and fit information.
        
        >>> chi2 = self.Calc_chi2([PIBYTWO,1.,0.9,4000.,0.08,300e3,5000.,10.,0.])
        """
        # We can provide a function that massages the input parameters and returns them.
        # This function can, for example, handle fixed parameters or boundary limits.
        if func_par is not None:
            par = func_par(par)
        # check if we are dealing with a dictionary
        if isinstance(par, dict):
            par = [par['incl'], par['corotation'], par['filling'], par['tnight'], par['gravdark'], par['k1'], par['tday'], par['dm'], par['av']]
        
        if offset_free == 0:
            pred_flux = self.Get_flux(par, flat=True, nsamples=nsamples, verbose=verbose)
            ((par[7],par[8]), chi2_data, rank, s) = Utils.Misc.Fit_linear(self.mag-pred_flux, x=self.ext, err=self.mag_err, b=par[7], m=par[8])
            if full_output:
                residuals = ( (self.mag-pred_flux) - (self.ext*par[8] + par[7]) ) / self.mag_err
                offset = np.zeros(self.ndataset)
            chi2_band = 0.
            chi2 = chi2_data + chi2_band
        else:
            # Calculate the theoretical flux
            pred_flux = self.Get_flux(par, flat=False, nsamples=nsamples, verbose=verbose)
            # Calculate the residuals between observed and theoretical flux
            if influx: # Calculate the residuals in the flux domain
                res1 = np.array([ Utils.Misc.Fit_linear(self.data['flux'][i], x=Utils.Flux.Mag_to_flux(pred_flux[i], flux0=self.atmo_grid[i].flux0), err=self.data['flux_err'][i], b=0., inline=True) for i in np.arange(self.ndataset) ])
                offset = -2.5*np.log10(res1[:,1])
                if full_output:
                    print( "Impossible to return proper residuals" )
                    residuals = None
            else: # Calculate the residuals in the magnitude domain
                res1 = np.array([ Utils.Misc.Fit_linear(self.data['mag'][i]-pred_flux[i], err=self.data['mag_err'][i], m=0., inline=True) for i in np.arange(self.ndataset) ])
                offset = res1[:,0]
                if full_output:
                    residuals = [ ((self.data['mag'][i]-pred_flux[i]) - offset[i])/self.data['mag_err'][i] for i in np.arange(self.ndataset) ]
            chi2_data = res1[:,2].sum()
            # Fit for the best offset between the observed and theoretical flux given the DM and A_V
            res2 = Utils.Misc.Fit_linear(offset, x=self.data['ext'], err=self.data['calib'], b=par[7], m=par[8], inline=True)
            par[7], par[8] = res2[0], res2[1]
            chi2_band = res2[2]
            # Here we add the chi2 of the data from that of the offsets for the bands.
            chi2 = chi2_data + chi2_band
            # Update the offset to be the actual offset between the data and the band (i.e. minus the DM and A_V contribution)
            offset -= self.data['ext']*par[8] + par[7]

        # Output results
        if verbose:
            print('chi2: {:.3f}, chi2 (data): {:.3f}, chi2 (band offset): {:.3f}, DM: {:.3f}, A_V: {:.3f}'.format(chi2, chi2_data, chi2_band, par[7], par[8]))
        if full_output:
            return chi2, {'offset':offset, 'par':par, 'res':residuals}
        else:
            return chi2

    def Get_flux(self, par, flat=False, func_par=None, DM_AV=False, nsamples=None, verbose=False):
        """Get_flux(par, flat=False, func_par=None, DM_AV=False, nsamples=None, verbose=False)
        Returns the predicted flux (in magnitude) by the model evaluated
        at the observed values in the data set.
        
        par: Parameter list.
            [0]: Orbital inclination in radians.
            [1]: Corotation factor.
            [2]: Roche-lobe filling.
            [3]: Companion temperature.
            [4]: Gravity darkening coefficient.
            [5]: K (projected velocity semi-amplitude) in m/s.
            [6]: Front side temperature or irradiation temperature.
                The irradiation temperature is in the case of the
                photometry_modeling_temperature class.
            [7]: Distance modulus (optional).
            [8]: Absorption A_V (optional).
            Note: Can also be a dictionary:
                par.keys() = ['av', 'corotation', 'dm', 'filling',
                    'gravdark', 'incl','k1','tday','tnight']
        flat (False): If True, the values are returned in a 1D vector.
            If False, predicted values are grouped by data set left in a list.
        func_par (None): Function that takes the parameter vector and
            returns the parameter vector. This allow for possible constraints
            on the parameters. The vector returned by func_par must have a length
            equal to the number of expected parameters.
        DM_AV (False): If true, will include the DM and A_V in the flux.
        nsamples (None): Number of points for the lightcurve sampling.
            If None, the lightcurve will be sampled at the observed data
            points.
        
        Note: tirr = (par[6]**4 - par[3]**4)**0.25
        
        >>> self.Get_flux([PIBYTWO,1.,0.9,4000.,0.08,300e3,5000.,10.,0.])
        """
        # func_par
        if func_par is not None:
            par = func_par(par)
        # check if we are dealing with a dictionary
        if isinstance(par, dict):
            par = [par['incl'], par['corotation'], par['filling'], par['tnight'], par['gravdark'], par['k1'], par['tday'], par['dm'], par['av']]
        
        # We call Make_surface to make the companion's surface.
        self.Make_surface(par, verbose=verbose)
        
        # If nsamples is None we evaluate the lightcurve at each data point.
        if nsamples is None:
            phases = self.data['phase']
        # If nsamples is set, we evaluate the lightcurve at nsamples
        else:
            phases = (np.arange(nsamples, dtype=float)/nsamples).repeat(self.ndataset).reshape((nsamples,self.ndataset)).T
        
        # If DM_AV, we take into account the DM and AV into the flux here.
        if DM_AV:
            DM_AV = self.data['ext']*par[8] + par[7]
        else:
            DM_AV = self.data['ext']*0.
        
        # Calculate the actual lightcurves
        flux = []
        for i in np.arange(self.ndataset):
                # If we use the interpolation method and if the filter is the same as a previously
                # calculated one, we do not recalculate the fluxes and simply copy them.
                if nsamples is not None and self.grouping[i] < i:
                    flux.append(flux[self.grouping[i]])
                else:
                    flux.append( np.array([self.star.Mag_flux(phase, atmo_grid=self.atmo_grid[i]) for phase in phases[i]]) + DM_AV[i] )
        
        # If nsamples is set, we interpolate the lightcurve at nsamples.
        if nsamples is not None:
            for i in np.arange(self.ndataset):
                ws, inds = Utils.Series.Getaxispos_vector(phases[i], self.data['phase'][i])
                flux[i] = flux[i][inds]*(1-ws) + flux[i][inds+1]*ws
        
        # We can flatten the flux array to simplify some of the calculations in the Calc_chi2 function
        if flat:
            return np.hstack(flux)
        else:
            return flux

    def Get_flux_theoretical(self, par, phases, func_par=None, verbose=False):
        """Get_flux_theoretical(par, phases, func_par=None, verbose=False)
        Returns the predicted flux (in magnitude) by the model evaluated at the
        observed values in the data set.
        
        par: Parameter list.
            [0]: Orbital inclination in radians.
            [1]: Corotation factor.
            [2]: Roche-lobe filling.
            [3]: Companion temperature.
            [4]: Gravity darkening coefficient.
            [5]: K (projected velocity semi-amplitude) in m/s.
            [6]: Front side temperature or irradiation temperature.
                The irradiation temperature is in the case of the
                photometry_modeling_temperature class.
            [7]: Distance modulus.
            [8]: Absorption A_V.
            Note: Can also be a dictionary:
                par.keys() = ['av','corotation','dm','filling','gravdark','incl','k1','tday','tnight']
        phases: A list of orbital phases at which the model should be
            evaluated. The list must have the same length as the
            number of data sets, each element can contain many phases.
        func_par (None): Function that takes the parameter vector and
            returns the parameter vector. This allow for possible constraints
            on the parameters. The vector returned by func_par must have a length
            equal to the number of expected parameters.
        verbose (False)
        
        Note: tirr = (par[6]**4 - par[3]**4)**0.25
        
        >>> self.Get_flux_theoretical([PIBYTWO,1.,0.9,4000.,0.08,300e3,5000.,10.,0.], [[0.,0.25,0.5,0.75]]*4)
        """
        # func_par
        if func_par is not None:
            par = func_par(par)
        # check if we are dealing with a dictionary
        if isinstance(par, dict):
            par = [par['incl'], par['corotation'], par['filling'], par['tnight'], par['gravdark'], par['k1'], par['tday'], par['dm'], par['av']]
        
        # We call Make_surface to make the companion's surface.
        self.Make_surface(par, verbose=verbose)
        
        DM_AV = self.data['ext']*par[8] + par[7]
        
        flux = []
        for i in np.arange(self.ndataset):
            # If the filter is the same as a previously calculated one
            # we do not recalculate the fluxes and simply copy them.
            if self.grouping[i] < i:
                flux.append( flux[self.grouping[i]] )
            else:
                flux.append( np.array([self.star.Mag_flux(phase, atmo_grid=self.atmo_grid[i]) for phase in phases[i]]) + DM_AV[i] )            
        return flux

    def Get_Keff(self, par, nphases=20, atmo_grid=0, func_par=None, make_surface=False, verbose=False):
        """
        Returns the effective projected velocity semi-amplitude of the star in m/s.
        The luminosity-weighted average velocity of the star is returned for
        nphases, for the specified dataset, and a sin wave is fitted to them.
        
        par: Parameter list.
            [0]: Orbital inclination in radians.
            [1]: Corotation factor.
            [2]: Roche-lobe filling.
            [3]: Companion temperature.
            [4]: Gravity darkening coefficient.
            [5]: K (projected velocity semi-amplitude) in m/s.
            [6]: Front side temperature.
            [7]: Distance modulus.
            [8]: Absorption A_V.
        nphases (int): Number of phases to evaluate the velocity at.
        atmo_grid (int, AtmoGridPhot): The atmosphere grid to use for the velocity
            calculation. Can be an integer that represents the index of the atmosphere
            grid object in self.atmo_grid, and it can be an AtmoGridPhot instance.
        func_par (function): Function that takes the parameter vector and
            returns the parameter vector. This allow for possible constraints
            on the parameters. The vector returned by func_par must have a length
            equal to the number of expected parameters.
        make_surface (bool): Whether lightcurve.make_surface should be called
            or not. If the flux has been evaluate before and the parameters have
            not changed, False is fine.
        verbose (bool): Verbosity. Will plot the velocities and the sin fit.
        """
        # If it is required to recalculate the stellar surface.
        if make_surface:
            self.Make_surface(par, func_par=func_par, verbose=verbose)
        # Deciding which atmosphere grid we use to evaluate Keff
        if isinstance(atmo_grid, int):
            atmo_grid = self.atmo_grid[atmo_grid]
        # Get the Keffs and fluxes
        phases = np.arange(nphases)/float(nphases)
        Keffs = np.array( [self.star.Keff(phase, atmo_grid=atmo_grid) for phase in phases] )
        tmp = Utils.Misc.Fit_linear(Keffs, np.sin(cts.TWOPI*(phases)), inline=True)
        if verbose:
            pylab.plot(np.linspace(0.,1.), tmp[1]*np.sin(np.linspace(0.,1.)*cts.TWOPI)+tmp[0])
            pylab.scatter(phases, Keffs)
        Keff = tmp[1]
        return Keff

    def _Init_lightcurve(self, ndiv, read=False):
        """_Init_lightcurve(ndiv, read=False)
        Call the appropriate Lightcurve class and initialize
        the stellar array.
        
        >>> self._Init_lightcurve(ndiv)
        """
        self.star = Core.Star(ndiv, read=read)
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
            [4]: Gravity darkening coefficient.
            [5]: K (projected velocity semi-amplitude) in m/s.
            [6]: Front side temperature or irradiation temperature.
                The irradiation temperature is in the case of the
                photometry_modeling_temperature class.
            [7]: Distance modulus (optional). Not needed here.
            [8]: Absorption A_V (optional). Not needed here.
            Note: Can also be a dictionary:
                par.keys() = ['av','corotation','dm','filling','gravdark','incl','k1','tday','tnight']
        func_par (None): Function that takes the parameter vector and
            returns the parameter vector. This allow for possible constraints
            on the parameters. The vector returned by func_par must have a length
            equal to the number of expected parameters.
        
        >>> self.Make_surface([PIBYTWO,1.,0.9,4000.,0.08,300e3,5000.,10.,0.])
        """
        # Apply a function that can modify the value of parameters.
        if func_par is not None:
            par = func_par(par)
        # check if we are dealing with a dictionary
        if isinstance(par, dict):
            par = [par['incl'], par['corotation'], par['filling'], par['tnight'], par['gravdark'], par['k1'], par['tday'], par['dm'], par['av']]
        
        # Verify parameter values to make sure they make sense.
        #if par[6] < par[3]: par[6] = par[3]
        # Let's move on with the flux calculation.
        q = par[5] * self.K_to_q
        tirr = (par[6]**4 - par[3]**4)**0.25
        
        if verbose:
            print( "#####\n" + str(par[0]) + ", " + str(par[1]) + ", " + str(par[2]) + ", " + str(par[3]) + ", " + str(par[4]) + ", " + str(par[5]) + ", " + str(par[6]) + ", " + str(par[7]) + ", " + str(par[8]) + "\n" + "q: " + str(q) + ", tirr: " + str(tirr)  )
        
        self.star.Make_surface(q=q, omega=par[1], filling=par[2], temp=par[3], tempgrav=par[4], tirr=tirr, porb=self.porb, k1=par[5], incl=par[0])
        return

    def Plot(self, par, nphases=51, verbose=True, func_par=None, nsamples=None, output=False):
        """
        Plots the observed and predicted values along with the
        light curve.
        
        par (list): Parameter list.
            [0]: Orbital inclination in radians.
            [1]: Corotation factor.
            [2]: Roche-lobe filling.
            [3]: Companion temperature.
            [4]: Gravity darkening coefficient.
            [5]: K (projected velocity semi-amplitude) in m/s.
            [6]: Front side temperature or irradiation temperature.
                The irradiation temperature is in the case of the
                photometry_modeling_temperature class.
            [7]: Distance modulus.
            [8]: Absorption A_V.
            Note: Can also be a dictionary:
                par.keys() = ['av','corotation','dm','filling','gravdark','incl','k1','tday','tnight']
        nphases (int): Orbital phase resolution of the model
            light curve.
        verbose (bool): verbosity.
        func_par (function): Function that takes the parameter vector and
            returns the parameter vector. This allow for possible constraints
            on the parameters. The vector returned by func_par must have a length
            equal to the number of expected parameters.
        nsamples (int): Number of points for the lightcurve sampling.
            If None, the lightcurve will be sampled at the observed data
            points.
        output (bool): If true, will return the model flux values and the offsets.
        
        >>> self.Plot([PIBYTWO,1.,0.9,4000.,0.08,300e3,5000.,10.,0.])
        """
        # Calculate the orbital phases at which the flux will be evaluated
        phases = np.resize(np.linspace(0.,1.,nphases), (self.ndataset, nphases))
        # Fit the data in order to get the offset
        chi2, extras = self.Calc_chi2(par, offset_free=1, verbose=verbose, func_par=func_par, nsamples=nsamples, full_output=True)
        offset = extras['offset']
        par = extras['par']
        # Calculate the theoretical flux at the orbital phases.
        pred_flux = self.Get_flux_theoretical(par, phases)
        # Calculating the min and the max
        tmp = []
        for i in np.arange(self.ndataset):
            tmp = np.r_[tmp, pred_flux[i]+offset[i]]
        minmag = tmp.min()
        maxmag = tmp.max()
        deltamag = (maxmag - minmag)
        spacing = 0.2

        #---------------------------------
        ##### Plot using matplotlib
        try:
            fig = pylab.gcf()
            try:
                ax = pylab.gca()
            except:
                ax = fig.add_subplot(1,1,1)
        except:
            fig, ax = pylab.subplots(nrows=1, ncols=1)
        ncolors = self.ndataset - 1
        if ncolors == 0:
            ncolors = 1
        for i in np.arange(self.ndataset):
            color = np.ones((self.data['mag'][i].size,1), dtype=float) * matplotlib.cm.jet(float(i)/ncolors)
            ax.errorbar(self.data['phase'][i], self.data['mag'][i], yerr=self.data['mag_err'][i], fmt='none', ecolor=color[0])
            ax.scatter(self.data['phase'][i], self.data['mag'][i], edgecolor=color, facecolor=color)
            ax.plot(phases[i], pred_flux[i], 'k--')
            ax.plot(phases[i], pred_flux[i]+offset[i], 'k-')
            ax.text(1.01, pred_flux[i].max(), self.data['id'][i])
        ax.set_xlim([0,1])
        ax.set_ylim([maxmag+spacing*deltamag, minmag-spacing*deltamag])
        ax.set_xlabel( "Orbital Phase" )
        ax.set_ylabel( "Magnitude" )
        pylab.draw()
        
        if output:
            return pred_flux, offset
        return

    def Plot_theoretical(self, par, nphases=31, verbose=False, device='/XWIN', func_par=None, output=False):
        """Plot_theoretical(par, nphases=31, verbose=False, device='/XWIN', func_par=None, output=False)
        Plots the predicted light curves.
        par: Parameter list.
            [0]: Orbital inclination in radians.
            [1]: Corotation factor.
            [2]: Roche-lobe filling.
            [3]: Companion temperature.
            [4]: Gravity darkening coefficient.
            [5]: K (projected velocity semi-amplitude) in m/s.
            [6]: Front side temperature or irradiation temperature.
                The irradiation temperature is in the case of the
                photometry_modeling_temperature class.
            [7]: Distance modulus.
            [8]: Absorption A_V.
            Note: Can also be a dictionary:
                par.keys() = ['av','corotation','dm','filling','gravdark','incl','k1','tday','tnight']
        nphases (31): Orbital phase resolution of the model
            light curve.
        verbose (False): verbosity.
        device ('/XWIN'): Device driver for Pgplot (can be '/XWIN',
            'filename.ps/PS', 'filename.ps./CPS', '/AQT' (on mac only)).
        func_par (None): Function that takes the parameter vector and
            returns the parameter vector. This allow for possible constraints
            on the parameters. The vector returned by func_par must have a length
            equal to the number of expected parameters.
        output (False): If true, will return the model flux values and the offsets.
        
        >>> self.Plot_theoretical([PIBYTWO,1.,0.9,4000.,0.08,300e3,5000.,10.,0.])
        """
        # Calculate the orbital phases at which the flux will be evaluated
        phases = np.resize(np.linspace(0.,1.,nphases), (self.ndataset, nphases))
        # Calculate the theoretical flux at the orbital phases.
        pred_flux = self.Get_flux_theoretical(par, phases, func_par=func_par, verbose=verbose)
        # Loop over the data set and plot the flux, theoretical flux and offset theoretical flux
        for i in np.arange(self.ndataset):
            plotxy(pred_flux[i], phases[i], color=1+i, line=1, rangey=[np.max(pred_flux)+0.5,np.min(pred_flux)-0.5], rangex=[0.,1.], device=device)
        if output:
            return pred_flux
        return

    def Pretty_print(self, par, make_surface=True, verbose=True):
        """Pretty_print(par, make_surface=True, verbose=True)
        Return a nice representation of the important
        parameters.
        
        par: Parameter list.
            [0]: Orbital inclination in radians.
            [1]: Corotation factor.
            [2]: Roche-lobe filling.
            [3]: Companion temperature.
            [4]: Gravity darkening coefficient.
            [5]: K (projected velocity semi-amplitude) in m/s.
            [6]: Front side temperature or irradiation temperature.
                The irradiation temperature is in the case of the
                photometry_modeling_temperature class.
            [7]: Distance modulus.
            [8]: Absorption A_V.
            Note: Can also be a dictionary:
                par.keys() = ['av','corotation','dm','filling','gravdark','incl','k1','tday','tnight']
        make_surface (True): Whether to recalculate the 
            surface of the star or not.
        verbose (True): Output the nice representation
            of the important parameters or just return them
            as a list.
        
        >>> self.Pretty_print([PIBYTWO,1.,0.9,4000.,0.08,300e3,5000.,10.,0.])
        """
        # check if we are dealing with a dictionary
        if isinstance(par, dict):
            par = [par['incl'], par['corotation'], par['filling'], par['tnight'], par['gravdark'], par['k1'], par['tday'], par['dm'], par['av']]
        
        incl = par[0]
        corot = par[1]
        fill = par[2]
        temp_back = par[3]
        gdark = par[4]
        K = par[5]
        temp_front = par[6]
        DM = par[7]
        A_V = par[8]
        if DM is None: DM = 0.
        if A_V is None: A_V = 0.
        q = K * self.K_to_q
        tirr = (temp_front**4 - temp_back**4)**0.25
        if make_surface:
            self.star.Make_surface(q=q, omega=corot, filling=fill, temp=temp_back, tempgrav=gdark, tirr=tirr, porb=self.porb, k1=K, incl=incl)
        separation = self.star.separation
        roche = self.star.Roche()
        Mwd = self.star.mass1
        Mns = self.star.mass2
        # below we transform sigma from W m^-2 K^-4 to erg s^-1 cm^-2 K^-4
        # below we transform the separation from m to cm
        Lirr = tirr**4 * (cts.sigma*1e3) * (separation*100)**2 * 4*cts.PI
        eff = Lirr/self.edot
        # we convert Lirr in Lsun units
        Lirr /= 3.839e33
        if verbose:
            print( "##### Pretty Print #####" )
            print( "%9.7f, %3.1f, %9.7f, %10.5f, %4.2f, %9.2f, %9.7f, %6.3f, %6.3f" %tuple(par) )
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
            print( "Frontside temperature: %7.2f (tabul.), %7.2f (approx.) K" %(np.exp(self.star.logteff.max()),temp_front) )
            print( "" )
            print( "Distance Modulus: %6.3f" %DM )
            print( "Absorption (V band): %6.3f" %A_V )
            print( "" )
            print( "Inclination: %5.3f rad (%6.2f deg)" %(incl,incl*cts.RADTODEG) )
            print( "K: %7.3f km/s" %(K/1000) )
            print( "" )
            print( "Mass ratio: %6.3f" %q )
            print( "Mass NS: %5.3f Msun" %Mns )
            print( "Mass Comp: %5.3f Msun" %Mwd )
        return np.r_[corot,gdark,fill,separation,roche,eff,tirr,temp_back,np.exp(self.star.logteff.max()),temp_front,DM,A_V,incl,incl*cts.RADTODEG,K,q,Mns,Mwd]

    def _Read_atmo(self, atmo_fln):
        """_Read_atmo(atmo_fln)
        Reads the atmosphere model data.
        
        atmo_fln (str): A file containing the grid model information for each
            data set. The format of each line of the file is as follows:
                Col 0: band name
                Col 1: band filename
        
        >>> self._Read_atmo(atmo_fln)
        """
        f = open(atmo_fln,'r')
        lines = f.readlines()
        self.atmo_grid = []
        for line in lines:
            if (line[0] != '#') and (line[0] != '\n'):
                tmp = line.split()
                self.atmo_grid.append(Atmosphere.AtmoGridPhot.ReadHDF5(tmp[1]))
        return

    def _Read_data(self, data_fln):
        """_Read_data(data_fln)
        Reads the photometric data.
        
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

        >>> self._Read_data(data_fln)
        """
        f = open(data_fln,'r')
        lines = f.readlines()
        self.data = {'phase':[], 'mag':[], 'mag_err':[], 'flux':[], 'flux_err':[], 'calib':[], 'fln':[], 'id':[], 'softening':[]}
        for line in lines:
            if (line[0] != '#') and (line[0] != '\n'):
                tmp = line.split()
                ## Old version of the data files
                if len(tmp) == 7:
                    d = np.loadtxt(tmp[-1], usecols=[int(tmp[1]),int(tmp[2]),int(tmp[3])], unpack=True)
                    ## With the flag '_' in the observation id, we do not take %1 so that
                    ## we preserve the long-term phase coherence.
                    if tmp[0].find('_') != -1:
                        self.data['phase'].append( np.atleast_1d(d[0] - float(tmp[4])) )
                    else:
                        self.data['phase'].append( np.atleast_1d((d[0] - float(tmp[4]))%1.) )
                    self.data['mag'].append( np.atleast_1d(d[1]) )
                    self.data['mag_err'].append( np.atleast_1d(d[2]) )
                    self.data['calib'].append( float(tmp[5]) )
                    self.data['fln'].append( tmp[-1] )
                    self.data['id'].append( tmp[0] )
                    self.data['softening'].append( 0. )
                ## Old version of the data files including asinh magnitudes
                elif len(tmp) == 8:
                    d = np.loadtxt(tmp[-1], usecols=[int(tmp[1]),int(tmp[2]),int(tmp[3])], unpack=True)
                    # With the flag '_' in the observation id, we do not take %1 so that
                    # we preserve the long-term phase coherence.
                    if tmp[0].find('_') != -1:
                        self.data['phase'].append( np.atleast_1d(d[0] - float(tmp[4])) )
                    else:
                        self.data['phase'].append( np.atleast_1d((d[0] - float(tmp[4]))%1.) )
                    self.data['mag'].append( np.atleast_1d(d[1]) )
                    self.data['mag_err'].append( np.atleast_1d(d[2]) )
                    self.data['calib'].append( float(tmp[5]) )
                    self.data['fln'].append( tmp[-1] )
                    self.data['id'].append( tmp[0] )
                    self.data['softening'].append( float(tmp[6]) )
                ## Current version of the data files including asinh magnitudes
                elif len(tmp) == 9:
                    d = np.loadtxt(tmp[-1], usecols=[int(tmp[1]),int(tmp[2]),int(tmp[3])], unpack=True)
                    ## Data can be set in magnitude
                    if tmp[-2] == 'mag':
                        # With the flag '_' in the observation id, we do not take %1 so that
                        # we preserve the long-term phase coherence.
                        if tmp[0].find('_') != -1:
                            self.data['phase'].append( np.atleast_1d(d[0] - float(tmp[4])) )
                        else:
                            self.data['phase'].append( np.atleast_1d((d[0] - float(tmp[4]))%1.) )
                        self.data['mag'].append( np.atleast_1d(d[1]) )
                        self.data['mag_err'].append( np.atleast_1d(d[2]) )
                        self.data['calib'].append( float(tmp[5]) )
                        self.data['fln'].append( tmp[-1] )
                        self.data['id'].append( tmp[0] )
                        self.data['softening'].append( float(tmp[6]) )
                    ## Data can be set in flux
                    elif tmp[-2] == 'flux':
                        # With the flag '_' in the observation id, we do not take %1 so that
                        # we preserve the long-term phase coherence.
                        if tmp[0].find('_') != -1:
                            self.data['phase'].append( np.atleast_1d(d[0] - float(tmp[4])) )
                        else:
                            self.data['phase'].append( np.atleast_1d((d[0] - float(tmp[4]))%1.) )
                        self.data['flux'].append( np.atleast_1d(d[1]) )
                        self.data['flux_err'].append( np.atleast_1d(d[2]) )
                        self.data['calib'].append( float(tmp[5]) )
                        self.data['fln'].append( tmp[-1] )
                        self.data['id'].append( tmp[0] )
                        self.data['softening'].append( float(tmp[6]) )
                ## Current version of the data files including asinh magnitudes
                else:
                    raise Exception("The data file does not have the expected number of columns.")
        return

    def _Setup(self):
        """_Setup()
        Stores some important information in class variables.
        
        >>> self._Setup()
        """
        # We calculate the constant for the conversion of K to q (observed
        # velocity semi-amplitude to mass ratio, with K in m/s)
        self.K_to_q = Utils.Binary.Get_K_to_q(self.porb, self.x2sini)
        # Storing values in 1D arrays.
        # The V band extinction will be extracted from the atmosphere_grid class
        ext = []
        self.data['ext'] = []
        # Converting magnitudes <-> fluxes in case this would be needed for upper limits
        if len(self.data['flux']) == 0:
            has_mag = True
        else:
            has_mag = False
        # The grouping will define datasets that are in the same band and can be evaluated only once in order to save on computation.
        grouping = np.arange(self.ndataset)
        for i in np.arange(self.ndataset):
            ext.extend(self.data['phase'][i]*0.+self.atmo_grid[i].meta['ext'])
            self.data['ext'].append(self.atmo_grid[i].meta['ext'])
            if self.data['softening'][i] == 0:
                if has_mag:
                    flux,flux_err = Utils.Flux.Mag_to_flux(self.data['mag'][i], mag_err=self.data['err'][i], flux0=self.atmo_grid[i].meta['zp'])
                    self.data['flux'].append( flux )
                    self.data['flux_err'].append( flux_err )
                else:
                    mag,mag_err = Utils.Flux.Flux_to_mag(self.data['flux'][i], flux_err=self.data['flux_err'][i], flux0=self.atmo_grid[i].meta['zp'])
                    self.data['mag'].append( mag )
                    self.data['mag_err'].append( mag_err )
            else:
                flux,flux_err = Utils.Flux.Asinh_to_flux(self.data['mag'][i], mag_err=self.data['mag_err'][i], flux0=self.atmo_grid[i].meta['zp'], softening=self.data['softening'][i])
                self.data['flux'].append( flux )
                self.data['flux_err'].append( flux_err )
            for j in np.arange(i+1):
                if self.data['id'][i] == self.data['id'][j]:
                    grouping[i] = j
                    break
        self.ext = np.asarray(ext)
        self.grouping = np.asarray(grouping)
        self.data['ext'] = np.asarray(self.data['ext'])
        self.data['calib'] = np.asarray(self.data['calib'])
        self.mag = np.hstack(self.data['mag'])
        self.mag_err = np.hstack(self.data['mag_err'])
        self.phase = np.hstack(self.data['phase'])
        self.flux = np.hstack(self.data['flux'])
        self.flux_err = np.hstack(self.data['flux_err'])
        self.ndata = self.flux.size
        return

######################## class Photometry ########################

