# Licensed under a 3-clause BSD style license - see LICENSE

__all__ = ["Photometry"]

from ..Utils.import_modules import *
from .. import Utils
from .. import Core
from .. import Atmosphere


######################## class Photometry ########################
class Photometry:
    """Photometry
    This class allows to fit the flux from the primary star
    of a binary system, assuming it is heated by the secondary
    (which in most cases will be a pulsar).
    
    It is meant to deal with photometry data. Many sets of photometry
    data (i.e. different filters) are read. For each data set, one can
    calculate the predicted flux of the model at every data point (i.e.
    for a given orbital phase).
    """
    def __init__(self, atmo_fln, data_fln, nalf, porb, x2sini, edot=1., read=True):
        """__init__(atmo_fln, data_fln, nalf, porb, x2sini, edot=1., read=True)
        This class allows to fit the flux from the primary star
        of a binary system, assuming it is heated by the secondary
        (which in most cases will be a pulsar).
        
        It is meant to deal with photometry data. Many sets of photometry
        data (i.e. different filters) are read. For each data set, one can
        calculate the predicted flux of the model at every data point (i.e.
        for a given orbital phase).
        
        atmo_fln (str): A file containing the grid model information for each
            data set. The format of each line of the file is as follows:
                band_name, center_wavelength, delta_wavelength, flux0,
                    extinction, grid_file
        data_fln (str): A file containing the information for each data set.
            The format of the file is as follows:
                band_name, column_phase, column_flux, column_error_flux,
                    shift_to_phase_zero, calibration_error, softening_asinh,
                    data_file
            Here, the first column has index 0.
            Here, orbital phase 0. is the superior conjunction of the pulsar.
        nalf (int): The number of surface slice. Defines how coarse/fine the
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
        
        >>> fit = Photometry(atmo_fln, data_fln, nalf, porb, x2sini)
        """
        # We define some class attributes.
        self.porb = porb
        self.x2sini = x2sini
        self.edot = edot
        # We read the data.
        self._Read_data(data_fln)
        # We read the atmosphere models with the atmo_grid class
        self._Read_atmo(atmo_fln)
        # We make sure that the length of data and atmo_dict are the same
        if len(self.atmo_grid) != len(self.data['mag']):
            print 'The number of atmosphere grids and data sets (i.e. photometric bands) do not match!!!'
            return
        else:
            # We keep in mind the number of datasets
            self.ndataset = len(self.atmo_grid)
        # We initialize some important class attributes.
        self._Init_lightcurve(nalf, read=read)
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
            [8]: Absorption A_J (can be None).
            Note: DM and A_J can be set to None. In which case, if
            offset_free = 1, these parameters will be fit for.
            Note: Can also be a dictionary:
                par.keys() = ['aj','corotation','dm','filling','gravdark','incl','k1','tday','tnight']
        offset_free (int):
            1) offset_free = 0:
                If the offset is not free and the DM and A_J are specified, the chi2
                is calculated directly without allowing an offset between the data and
                the bands.
                The full chi2 should be:
                    chi2 = sum[ w_i*(off_i-dm-aj*C_i)**2]
                        + w_dm*(dm-dm_obs)**2 
                        + w_aj*(aj-aj_obs)**2,     with w = 1/sigma**2
                The extra terms (i.e. dm-dm_obs and aj-aj_obs) should be included
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
                values of DM and AJ.
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
            par = [par['incl'], par['corotation'], par['filling'], par['tnight'], par['gravdark'], par['k1'], par['tday'], par['dm'], par['aj']]
        
        if offset_free == 0:
            pred_flux = self.Get_flux(par, flat=True, nsamples=nsamples, verbose=verbose)
            ((par[7],par[8]), chi2_data, rank, s) = Utils.Fit_linear(self.mag-pred_flux, x=self.ext, err=self.err, b=par[7], m=par[8])
            if full_output:
                residuals = ( (self.mag-pred_flux) - (self.ext*par[8] + par[7]) ) / self.err
                offset = numpy.zeros(self.ndataset)
            chi2_band = 0.
            chi2 = chi2_data + chi2_band
        else:
            # Calculate the theoretical flux
            pred_flux = self.Get_flux(par, flat=False, nsamples=nsamples, verbose=verbose)
            # Calculate the residuals between observed and theoretical flux
            if influx: # Calculate the residuals in the flux domain
                res1 = numpy.array([ Utils.Fit_linear(self.data['flux'][i], x=Utils.Mag_to_flux(pred_flux[i], flux0=self.atmo_grid[i].flux0), err=self.data['flux_err'][i], b=0., inline=True) for i in numpy.arange(self.ndataset) ])
                offset = -2.5*numpy.log10(res1[:,1])
                if full_output:
                    print( "Impossible to return proper residuals" )
                    residuals = None
            else: # Calculate the residuals in the magnitude domain
                res1 = numpy.array([ Utils.Fit_linear(self.data['mag'][i]-pred_flux[i], err=self.data['err'][i], m=0., inline=True) for i in numpy.arange(self.ndataset) ])
                offset = res1[:,0]
                if full_output:
                    residuals = [ ((self.data['mag'][i]-pred_flux[i]) - offset[i])/self.data['err'][i] for i in numpy.arange(self.ndataset) ]
            chi2_data = res1[:,2].sum()
            # Fit for the best offset between the observed and theoretical flux given the DM and A_J
            res2 = Utils.Fit_linear(offset, x=self.data['ext'], err=self.data['calib'], b=par[7], m=par[8], inline=True)
            par[7], par[8] = res2[0], res2[1]
            chi2_band = res2[2]
            # Here we add the chi2 of the data from that of the offsets for the bands.
            chi2 = chi2_data + chi2_band
            # Update the offset to be the actual offset between the data and the band (i.e. minus the DM and AJ contribution)
            offset -= self.data['ext']*par[8] + par[7]

        # Output results
        if verbose:
            print('chi2: {:.3f}, chi2 (data): {:.3f}, chi2 (band offset): {:.3f}, D.M.: {:.3f}, AJ: {:.3f}'.format(chi2, chi2_data, chi2_band, par[7], par[8]))
        if full_output:
            return chi2, {'offset':offset, 'par':par, 'res':residuals}
        else:
            return chi2

    def Get_flux(self, par, flat=False, func_par=None, DM_AJ=False, nsamples=None, verbose=False):
        """Get_flux(par, flat=False, func_par=None, DM_AJ=False, nsamples=None, verbose=False)
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
            [8]: Absorption A_J (optional).
            Note: Can also be a dictionary:
                par.keys() = ['aj','corotation','dm','filling','gravdark','incl','k1','tday','tnight']
        flat (False): If True, the values are returned in a 1D vector.
            If False, predicted values are grouped by data set left in a list.
        func_par (None): Function that takes the parameter vector and
            returns the parameter vector. This allow for possible constraints
            on the parameters. The vector returned by func_par must have a length
            equal to the number of expected parameters.
        DM_AJ (False): If true, will include the DM and AJ in the flux.
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
            par = [par['incl'], par['corotation'], par['filling'], par['tnight'], par['gravdark'], par['k1'], par['tday'], par['dm'], par['aj']]
        
        # We call _Make_surface to make the companion's surface.
        self._Make_surface(par, verbose=verbose)
        
        # If nsamples is None we evaluate the lightcurve at each data point.
        if nsamples is None:
            phases = self.data['phase']
        # If nsamples is set, we evaluate the lightcurve at nsamples
        else:
            phases = (numpy.arange(nsamples, dtype=float)/nsamples).repeat(self.ndataset).reshape((nsamples,self.ndataset)).T
        
        # If DM_AJ, we take into account the DM and AJ into the flux here.
        if DM_AJ:
            DM_AJ = self.data['ext']*par[8] + par[7]
        else:
            DM_AJ = self.data['ext']*0.
        
        # Calculate the actual lightcurves
        flux = []
        for i in numpy.arange(self.ndataset):
                # If we use the interpolation method and if the filter is the same as a previously
                # calculated one, we do not recalculate the fluxes and simply copy them.
                if nsamples is not None and self.grouping[i] < i:
                    flux.append(flux[self.grouping[i]])
                else:
                    flux.append( numpy.array([self.lightcurve.Mag_flux(phase, atmo_grid=self.atmo_grid[i]) for phase in phases[i]]) + DM_AJ[i] )
        
        # If nsamples is set, we interpolate the lightcurve at nsamples.
        if nsamples is not None:
            for i in numpy.arange(self.ndataset):
                ws, inds = Utils.Getaxispos_vector(phases[i], self.data['phase'][i])
                flux[i] = flux[i][inds]*(1-ws) + flux[i][inds+1]*ws
        
        # We can flatten the flux array to simplify some of the calculations in the Calc_chi2 function
        if flat:
            return numpy.hstack(flux)
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
            [8]: Absorption A_J.
            Note: Can also be a dictionary:
                par.keys() = ['aj','corotation','dm','filling','gravdark','incl','k1','tday','tnight']
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
            par = [par['incl'], par['corotation'], par['filling'], par['tnight'], par['gravdark'], par['k1'], par['tday'], par['dm'], par['aj']]
        
        # We call _Make_surface to make the companion's surface.
        self._Make_surface(par, verbose=verbose)
        
        DM_AJ = self.data['ext']*par[8] + par[7]
        
        flux = []
        for i in numpy.arange(self.ndataset):
            # If the filter is the same as a previously calculated one
            # we do not recalculate the fluxes and simply copy them.
            if self.grouping[i] < i:
                flux.append( flux[self.grouping[i]] )
            else:
                flux.append( numpy.array([self.lightcurve.Mag_flux(phase, atmo_grid=self.atmo_grid[i]) for phase in phases[i]]) + DM_AJ[i] )            
        return flux

    def Get_Keff(self, par, nphases=20, dataset=None, func_par=None, make_surface=False, verbose=False):
        """Get_Keff(par, phases, dataset=None, func_par=None, make_surface=False, verbose=False)
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
            [8]: Absorption A_J.
        nphases (int): Number of phases to evaluate the velocity at.
        dataset (int): The dataset for which the velocity is evaluated
            (i.e. the atmosphere grid to use).
            This parameter must be set if not atmosphere grid was specified
            for the Keff evaluation in the class initialization.
        func_par (function): Function that takes the parameter vector and
            returns the parameter vector. This allow for possible constraints
            on the parameters. The vector returned by func_par must have a length
            equal to the number of expected parameters.
        make_surface (bool): Whether lightcurve.make_surface should be called
            or not. If the flux has been evaluate before and the parameters have
            not changed, False is fine.
        verbose (bool): Verbosity. Will plot the velocities and the sin fit.
        """
        # Apply a function that can modify the value of parameters.
        if func_par is not None:
            par = func_par(par)
        # If it is required to recalculate the stellar surface.
        if make_surface:
            q = par[5] * self.K_to_q
            tirr = (par[6]**4 - par[3]**4)**0.25
            self.lightcurve.Make_surface(q=q, omega=par[1], filling=par[2], temp=par[3], tempgrav=par[4], tirr=tirr, porb=self.porb, k1=par[5], incl=par[0])
        # Deciding which atmosphere grid we use to evaluate Keff
        if dataset is None:
            try:
                atmo_grid = self.keff_atmo_grid
            except:
                atmo_grid = self.atmo_grid[0]
        else:
            atmo_grid = self.atmo_grid[dataset]
        # Get the Keffs and fluxes
        phases = numpy.arange(nphases)/float(nphases)
        Keffs = numpy.array( [self.lightcurve.Keff(phase, atmo_grid=atmo_grid) for phase in phases] )
        tmp = Utils.Fit_linear(-Keffs, numpy.sin(TWOPI*(phases)), inline=True)
        if verbose:
            plotxy(-tmp[1]*numpy.sin(numpy.linspace(0.,1.)*TWOPI)+tmp[0], numpy.linspace(0.,1.))
            plotxy(Keffs, phases, line=None, symbol=2)
        Keff = tmp[1]
        return Keff

    def _Init_lightcurve(self, nalf, read=False):
        """_Init_lightcurve(nalf, read=False)
        Call the appropriate Lightcurve class and initialize
        the stellar array.
        
        >>> self._Init_lightcurve(nalf)
        """
        self.lightcurve = Core.Star(nalf, read=read)
        return

    def _Make_surface(self, par, func_par=None, verbose=False):
        """_Make_surface(par, func_par=None, verbose=False)
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
            [8]: Absorption A_J (optional). Not needed here.
            Note: Can also be a dictionary:
                par.keys() = ['aj','corotation','dm','filling','gravdark','incl','k1','tday','tnight']
        func_par (None): Function that takes the parameter vector and
            returns the parameter vector. This allow for possible constraints
            on the parameters. The vector returned by func_par must have a length
            equal to the number of expected parameters.
        
        >>> self._Make_surface([PIBYTWO,1.,0.9,4000.,0.08,300e3,5000.,10.,0.])
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
        tirr = (par[6]**4 - par[3]**4)**0.25
        
        if verbose:
            print( "#####\n" + str(par[0]) + ", " + str(par[1]) + ", " + str(par[2]) + ", " + str(par[3]) + ", " + str(par[4]) + ", " + str(par[5]) + ", " + str(par[6]) + ", " + str(par[7]) + ", " + str(par[8]) + "\n" + "q: " + str(q) + ", tirr: " + str(tirr)  )
        
        self.lightcurve.Make_surface(q=q, omega=par[1], filling=par[2], temp=par[3], tempgrav=par[4], tirr=tirr, porb=self.porb, k1=par[5], incl=par[0])
        return

    def Plot(self, par, nphases=31, verbose=True, device='/XWIN', func_par=None, nsamples=None, output=False):
        """Plot(par, nphases=31, verbose=True, device='/XWIN', func_par=None, nsamples=None, output=False)
        Plots the observed and predicted values along with the
        light curve.
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
            [8]: Absorption A_J.
            Note: Can also be a dictionary:
                par.keys() = ['aj','corotation','dm','filling','gravdark','incl','k1','tday','tnight']
        nphases (31): Orbital phase resolution of the model
            light curve.
        verbose (True): verbosity.
        device ('/XWIN'): Device driver for Pgplot (can be '/XWIN',
            'filename.ps/PS', 'filename.ps./CPS', '/AQT' (on mac only)).
        func_par (None): Function that takes the parameter vector and
            returns the parameter vector. This allow for possible constraints
            on the parameters. The vector returned by func_par must have a length
            equal to the number of expected parameters.
        nsamples (None): Number of points for the lightcurve sampling.
            If None, the lightcurve will be sampled at the observed data
            points.
        output (False): If true, will return the model flux values and the offsets.
        
        >>> self.Plot([PIBYTWO,1.,0.9,4000.,0.08,300e3,5000.,10.,0.])
        """
        # Calculate the orbital phases at which the flux will be evaluated
        phases = numpy.resize(numpy.linspace(0.,1.,nphases), (self.ndataset, nphases))
        # Fit the data in order to get the offset
        chi2, extras = self.Calc_chi2(par, offset_free=1, verbose=verbose, func_par=func_par, nsamples=nsamples, full_output=True)
        offset = extras['offset']
        par = extras['par']
        # Calculate the theoretical flux at the orbital phases.
        pred_flux = self.Get_flux_theoretical(par, phases)
        # Loop over the data set and plot the flux, theoretical flux and offset theoretical flux
        for i in numpy.arange(self.ndataset):
            plotxy(self.data['mag'][i], self.data['phase'][i], erry=self.data['err'][i], line=None, symbol=1, color=1+i, rangey=[self.mag.max()+0.5,self.mag.min()-0.5], rangex=[0.,1.], device=device)
            plotxy(pred_flux[i], phases[i], color=1+i, line=2)
            #plotxy(pred_flux[i]+offset[i], phases[i], color=1+i)
            plotxy(pred_flux[i]+offset[i], phases[i], color=1)
        plotxy([0],[0], color=1)
        y0 = (self.mag.max()+self.mag.min())/2
        dy = (self.mag.max()-self.mag.min())/25
        # Displaying information about the parameters
        ppgplot.pgsch(0.7) # Make the font smaller
        ppgplot.pgtext(0.4, y0+0*dy, 'Incl.: %4.2f deg'%(par[0]*RADTODEG))
        ppgplot.pgtext(0.4, y0+1*dy, 'Co-rot.: %3.1f'%par[1])
        ppgplot.pgtext(0.4, y0+2*dy, 'Fill.: %5.3f'%par[2])
        ppgplot.pgtext(0.4, y0+3*dy, 'Grav.: %4.2f'%par[4])
        if ( type(par[3]) == type([]) ) or ( type(par[3]) == type(numpy.array([])) ):
            ppgplot.pgtext(0.4, y0+4*dy, 'Temp. back: '+' ,'.join("%7.2f"%s for s in par[3])+' K')
        else:
            ppgplot.pgtext(0.4, y0+4*dy, 'Temp. back: %7.2f K'%par[3])
        ppgplot.pgtext(0.4, y0+5*dy, 'Temp. front: %7.2f K'%par[6])
        # in the following, we divide the speed by 1000 to convert from m/s to km/s
        ppgplot.pgtext(0.4, y0+6*dy, 'K: %5.2f km/s'%(par[5]/1000))
        ppgplot.pgtext(0.4, y0+7*dy, 'D.M.: %4.2f'%par[7])
        ppgplot.pgtext(0.4, y0+8*dy, 'Aj: %4.2f'%par[8])
        ppgplot.pgtext(0.4, y0+9.5*dy, 'q: %5.3f'%self.lightcurve.q)
        ppgplot.pgtext(0.4, y0+11*dy, 'Chi2: %7.2f, d.o.f.: %i'%(chi2,self.mag.size-len(par)))
        ppgplot.pgsch(1.0) # Restore the font size
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
            [8]: Absorption A_J.
            Note: Can also be a dictionary:
                par.keys() = ['aj','corotation','dm','filling','gravdark','incl','k1','tday','tnight']
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
        phases = numpy.resize(numpy.linspace(0.,1.,nphases), (self.ndataset, nphases))
        # Calculate the theoretical flux at the orbital phases.
        pred_flux = self.Get_flux_theoretical(par, phases, func_par=func_par, verbose=verbose)
        # Loop over the data set and plot the flux, theoretical flux and offset theoretical flux
        for i in numpy.arange(self.ndataset):
            plotxy(pred_flux[i], phases[i], color=1+i, line=1, rangey=[numpy.max(pred_flux)+0.5,numpy.min(pred_flux)-0.5], rangex=[0.,1.], device=device)
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
            [8]: Absorption A_J.
            Note: Can also be a dictionary:
                par.keys() = ['aj','corotation','dm','filling','gravdark','incl','k1','tday','tnight']
        make_surface (True): Whether to recalculate the 
            surface of the star or not.
        verbose (True): Output the nice representation
            of the important parameters or just return them
            as a list.
        
        >>> self.Pretty_print([PIBYTWO,1.,0.9,4000.,0.08,300e3,5000.,10.,0.])
        """
        # check if we are dealing with a dictionary
        if isinstance(par, dict):
            par = [par['incl'], par['corotation'], par['filling'], par['tnight'], par['gravdark'], par['k1'], par['tday'], par['dm'], par['aj']]
        
        incl = par[0]
        corot = par[1]
        fill = par[2]
        temp_back = par[3]
        gdark = par[4]
        K = par[5]
        temp_front = par[6]
        DM = par[7]
        A_J = par[8]
        if DM is None: DM = 0.
        if A_J is None: A_J = 0.
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
            print( "Frontside temperature: %7.2f (tabul.), %7.2f (approx.) K" %(numpy.exp(self.lightcurve.logteff.max()),temp_front) )
            print( "" )
            print( "Distance Modulus: %6.3f" %DM )
            print( "Absorption (J band): %6.3f" %A_J )
            print( "" )
            print( "Inclination: %5.3f rad (%6.2f deg)" %(incl,incl*RADTODEG) )
            print( "K: %7.3f km/s" %(K/1000) )
            print( "" )
            print( "Mass ratio: %6.3f" %q )
            print( "Mass NS: %5.3f Msun" %Mns )
            print( "Mass Comp: %5.3f Msun" %Mwd )
        return numpy.r_[corot,gdark,fill,separation,roche,eff,tirr,temp_back,numpy.exp(self.lightcurve.logteff.max()),temp_front,DM,A_J,incl,incl*RADTODEG,K,q,Mns,Mwd]

    def _Read_atmo(self, atmo_fln):
        """_Read_atmo(atmo_fln)
        Reads the atmosphere model data.
        
        atmo_fln: A file containing the grid model information for each
            data set. The format of each line of the file is as follows:
                band_name, center_wavelength, delta_wavelength, flux0,
                    extinction, grid_file
        
        >>> self._Read_atmo(atmo_fln)
        """
        f = open(atmo_fln,'r')
        lines = f.readlines()
        self.atmo_grid = []
        for line in lines:
            if (line[0] != '#') and (line[0] != '\n'):
                tmp = line.split()
                # We read the BT-Settl.7 data with the proper class
                if tmp[5].find('BT-Settl.7') != -1:
                    self.atmo_grid.append(Atmosphere.Atmo_grid_BTSettl7(tmp[5], float(tmp[1]), float(tmp[2]), float(tmp[3]), float(tmp[4])))
                else:
                    self.atmo_grid.append(Atmosphere.Atmo_grid(tmp[5], float(tmp[1]), float(tmp[2]), float(tmp[3]), float(tmp[4])))
            elif (line[:2] == '#!'):
                tmp = line.split()
                tmp = tmp[1:]
                if tmp[5].find('BT-Settl.7') != -1:
                    self.keff_atmo_grid = Atmosphere.Atmo_grid_BTSettl7(tmp[5], float(tmp[1]), float(tmp[2]), float(tmp[3]), float(tmp[4]))
                else:
                    self.keff_atmo_grid = Atmosphere.Atmo_grid(tmp[5], float(tmp[1]), float(tmp[2]), float(tmp[3]), float(tmp[4]))
        return

    def _Read_data(self, data_fln):
        """_Read_data(data_fln)
        Reads the photometric data.
        
        data_fln: A file containing the information for each data set.
            The format of the file is as follows:
                band_name, column_phase, column_flux, column_error_flux,
                    shift_to_phase_zero, calibration_error, softening_asinh,
                    data_file
        
        The softening_asinh parameter will be used for the conversion of the
        magnitudes in the data file into fluxes. If the value of 0. is provided,
        then the regular magnitudes are used instead of the asinh magnitudes.
        
        
        >>> self._Read_data(data_fln)
        """
        f = open(data_fln,'r')
        lines = f.readlines()
        self.data = {'mag':[], 'phase':[], 'err':[], 'calib':[], 'fln':[], 'id':[], 'softening':[]}
        for line in lines:
            if (line[0] != '#') and (line[0] != '\n'):
                tmp = line.split()
                d = numpy.loadtxt(tmp[-1], usecols=[int(tmp[1]),int(tmp[2]),int(tmp[3])], unpack=True)
                # With the flag '_' in the observation id, we do not take %1 so that
                # we preserve the long-term phase coherence.
                if tmp[0].find('_') != -1:
                    self.data['phase'].append( numpy.atleast_1d(d[0] - float(tmp[4])) )
                else:
                    self.data['phase'].append( numpy.atleast_1d((d[0] - float(tmp[4]))%1.) )
                self.data['mag'].append( numpy.atleast_1d(d[1]) )
                self.data['err'].append( numpy.atleast_1d(d[2]) )
                self.data['calib'].append( float(tmp[5]) )
                self.data['fln'].append( tmp[-1] )
                self.data['id'].append( tmp[0] )
                # If we use the new file format, there will be 8 columns. Number 6 will be for the asinh softening parameter.
                if len(tmp) == 8:
                    self.data['softening'].append( float(tmp[6]) )
                else:
                    self.data['softening'].append( 0. )
        return

    def _Setup(self):
        """_Setup()
        Stores some important information in class variables.
        
        >>> self._Setup()
        """
        # We calculate the constant for the conversion of K to q (observed
        # velocity semi-amplitude to mass ratio, with K in m/s)
        self.K_to_q = Utils.Get_K_to_q(self.porb, self.x2sini)
        # Storing values in 1D arrays.
        # The J band extinction will be extracted from the atmosphere_grid class
        ext = []
        self.data['ext'] = []
        # Converting magnitudes in fluxes in case this would be needed for upper limits
        self.data['flux'] = []
        self.data['flux_err'] = []
        # The grouping will define datasets that are in the same band and can be evaluated only once in order to save on computation.
        grouping = numpy.arange(self.ndataset)
        for i in numpy.arange(self.ndataset):
            ext.extend(self.data['phase'][i]*0.+self.atmo_grid[i].ext)
            self.data['ext'].append(self.atmo_grid[i].ext)
            if self.data['softening'][i] == 0:
                flux,flux_err = Utils.Mag_to_flux(self.data['mag'][i], mag_err=self.data['err'][i], flux0=self.atmo_grid[i].flux0)
            else:
                flux,flux_err = Utils.Asinh_to_flux(self.data['mag'][i], mag_err=self.data['err'][i], flux0=self.atmo_grid[i].flux0, softening=self.data['softening'][i])
            self.data['flux'].append( flux )
            self.data['flux_err'].append( flux_err )
            for j in numpy.arange(i+1):
                if self.data['id'][i] == self.data['id'][j]:
                    grouping[i] = j
                    break
        self.ext = numpy.array(ext)
        self.grouping = numpy.array(grouping)
        self.data['ext'] = numpy.array(self.data['ext'])
        self.data['calib'] = numpy.array(self.data['calib'])
        self.mag = numpy.hstack(self.data['mag'])
        self.err = numpy.hstack(self.data['err'])
        self.phase = numpy.hstack(self.data['phase'])
        self.flux = numpy.hstack(self.data['flux'])
        self.flux_err = numpy.hstack(self.data['flux_err'])
        self.ndata = self.flux.size
        return

######################## class Photometry ########################

