# Licensed under a 3-clause BSD style license - see LICENSE

__all__ = ["Photometry_disk"]

from ..Utils.import_modules import *
from .. import Utils
from .. import Core
from .. import Atmosphere


######################## class Photometry_disk ########################
class Photometry_disk(object):
    """Photometry_disk
    This class allows to fit the flux from the primary star
    of a binary system, assuming it is heated by the secondary
    (which in most cases will be a pulsar).
    It also handles a disk surrounding the pulsar, approximated
    as a constant flux contribution to the total flux.
    
    It is meant to deal with photometry data. Many sets of photometry
    data (i.e. different filters) are read. For each data set, one can
    calculate the predicted flux of the model at every data point (i.e.
    for a given orbital phase).
    """
    def __init__(self, atmo_fln, data_fln, ndiv, porb, x2sini, edot=1., DM=0., DMerr=0., AV=0., AVerr=0., Keff=0., Kefferr=0., read=False):
        """__init__(atmo_fln, data_fln, ndiv, porb, x2sini, edot=1., DM=0., DMerr=0., AV=0., AVerr=0., Keff=0., Kefferr=0., read=False)
        This class allows to fit the flux from the primary star
        of a binary system, assuming it is heated by the secondary
        (which in most cases will be a pulsar).
        It also handles a disk surrounding the pulsar, approximated
        as a constant flux contribution to the total flux.
        
        It is meant to deal with photometry data. Many sets of photometry
        data (i.e. different filters) are read. For each data set, one can
        calculate the predicted flux of the model at every data point (i.e.
        for a given orbital phase).
                
        atmo_fln: A file containing the grid model information for each
            data set. The format of each line of the file is as follows:
                band_name, center_wavelength, delta_wavelength, flux0,
                    extinction, grid_file
            Lines with '#' are comments and not read, except if '#!'. In
            which case, it means that the line should be use to set the
            atmosphere grid to be used when calculating Keff.

        data_fln: A file containing the information for each data set.
            The format of the file is as follows:
                band_name, column_phase, column_flux, column_error_flux,
                    shift_to_phase_zero, calibration_error, data_file
            Lines with '#' are comments and not read.
        ndiv: The number of surface element subdivisions. Defines how
            coarse/fine the surface grid is.
        porb: Orbital period of the system in seconds.
        x2sini: Projected semi-major axis of the secondary (pulsar)
            in light-second.
        edot (1.): Irradiated energy from the secondary, aka pulsar (i.e.
            spin-down luminosity) in erg/s. This is only used for the
            calculation of the irradiation efficiency so it does not
            enter in the modeling itself.
        DM (0.): Distance Modulus to the source, if known.
        DMerr (0.): Reciprocal of the Distance Modulus error to the source,
            if known. A value of 0. will disable the chi2DM contribution.
        AV (0.): V band absorption to the source, if known.
        AVerr (0.): Reciprocal of the V band absorption error to the source,
            if known. A value of 0. will disable the chi2AV contribution.
        Keff (0.): Effective projected velocity semi-amplitude in m/s, if known.
        Kefferr (0.): Reciprocal of the effective projected velocity
            semi-amplitude in m/s, if known.
        read (False): Whether the geodesic surface should be read from a file or
            generated from scratch.
        
        >>> fit = Photometry(atmo_fln, data_fln, ndiv, porb, x2sini, edot)
        """
        # We define some class attributes.
        self.porb = porb
        self.x2sini = x2sini
        self.edot = edot
        self.DM = DM
        self.DMerr = DMerr
        self.AV = AV
        self.AVerr = AVerr
        self.Keff = Keff
        self.Kefferr = Kefferr
        # We read the data.
        self.__Read_data(data_fln)
        # We read the atmosphere models with the atmo_grid class
        self.__Read_atmo(atmo_fln)
        # We make sure that the length of data and atmo_dict are the same
        if len(self.atmo_grid) != len(self.data['mag']):
            print 'The number of atmosphere grids and data sets (i.e. photometric bands) do not match!!!'
            return
        else:
            # We keep in mind the number of datasets
            self.ndataset = len(self.atmo_grid)
        # We initialize some important class attributes.
        self.star = Core.Star_disk(ndiv, read=read)
        self.__Setup()

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
                values of DM and AV.
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
            #pred_flux = self.Get_flux(par, flat=False, nsamples=nsamples, verbose=verbose)
            pred_flux = self.Get_flux(par, flat=False, verbose=verbose)
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
                    residuals = np.r_[ [ ((self.data['mag'][i]-pred_flux[i]) - offset[i])/self.data['mag_err'][i] for i in np.arange(self.ndataset) ] ]
            chi2_data = res1[:,2].sum()
            # Fit for the best offset between the observed and theoretical flux given the DM and A_V
            res2 = Utils.Misc.Fit_linear(offset, x=self.data['ext'], err=self.data['calib'], b=par[7], m=par[8], inline=True)
            par[7], par[8] = res2[0], res2[1]
            chi2_band = res2[2]
            # Here we add the chi2 of the data from that of the offsets for the bands.
            chi2 = chi2_data + chi2_band
            # Update the offset to be the actual offset between the data and the band (i.e. minus the DM and AV contribution)
            offset -= self.data['ext']*par[8] + par[7]

        # Output results
        if verbose:
            print('chi2: {:.3f}, chi2 (data): {:.3f}, chi2 (band offset): {:.3f}, D.M.: {:.3f}, AV: {:.3f}'.format(chi2, chi2_data, chi2_band, par[7], par[8]))
        if full_output:
            return chi2, {'offset':offset, 'par':par, 'res':residuals}
        else:
            return chi2

    def Calc_chi2_disk(self, par, func_par=None, offset_free=1, doKeff=False, return_residuals=False, return_disk=False, verbose=False):
        """Calc_chi2_disk(par, func_par=None, offset_free=1, doKeff=False, return_residuals=False, return_disk=False, verbose=False)
        Returns the chi-square of the fit of the data to the model.
        
        par: Parameter list.
            [0]: Orbital inclination in radians.
            [1]: Corotation factor.
            [2]: Roche-lobe filling.
            [3]: Companion temperature.
            [4]: Gravity darkening coefficient.
            [5]: K (projected velocity semi-amplitude) in m/s.
            [6]: Front side temperature.
            [7]: Distance modulus (can be None).
            [8]: Absorption A_V (can be None).
            [9-?]: Disk flux.
            Note: If there extra parameters after [9], they are assumed to
            be the individual disk fluxes of each data set.
            Note: Unlike Calc_chi2, DM and A_V cannot be fitted for.
        func_par (None): Function that takes the parameter vector and
            returns the parameter vector. This allow for possible constraints
            on the parameters. The vector returned by func_par must have a length
            equal to the number of expected parameters.
        offset_free (1):
            1) offset_free = 1:
            A fit is performed in order to find the best disk contribution.
            2) offset_free = 0:
            If the offset is not free all values are used directly for the
            calculation.
        doKeff (False): If true, will calculate the effective velocity
            semi-amplitude and will use the class variable Keff and Kefferr
            as a constraint to be added to the chi2.
        return_residuals (False): If true, will return a vector of residuals instead.
            (sqrt(chi2) in fact, so not exactly residuals).
        return_disk (False): If true, returns the disk flux and slope along with
            the chi2. Note that this option doesn't work with return_residuals.
        verbose (False): If true will display the list of parameters.
        
        >>> chi2 = self.Calc_chi2_disk([PIBYTWO,1.,0.9,4000.,0.08,1.4,0.07,10.,0.])
        >>> chi2, disk, disk_slope = self.Calc_chi2_disk([PIBYTWO,1.,0.9,4000.,0.08,1.4,0.07,10.,0.], return_disk=True)
        """
        # We can provide a function that massages the input parameters and returns them.
        # This function can, for example, handle fixed parameters or boundary limits.
        if func_par is not None:
            par = func_par(par)
        
        # Calculate the theoretical flux, not in magnitude but in flux!!!
        q = par[5] * self.K_to_q
        tirr = (par[6]**4 - par[3]**4)**0.25
        if verbose:
            print( "#####\n" + str(par[0]) + ", " + str(par[1]) + ", " + str(par[2]) + ", " + str(par[3]) + ", " + str(par[4]) + ", " + str(par[5]) + ", " + str(par[6]) + ", " + str(par[7]) + ", " + str(par[8]) + ", " + str(par[9]) + "\n" + "q: " + str(q) + ", tirr: " + str(tirr)  )
        self.star.Make_surface(q=q, omega=par[1], filling=par[2], temp=par[3], tempgrav=par[4], tirr=tirr, porb=self.porb, k1=par[5], incl=par[0])
        pred_flux = [np.array([self.star.Flux_disk(phase, atmo_grid=self.atmo_grid[i], disk=0.) for phase in self.data['phase'][i]]) for i in np.arange(self.ndataset)]
        
        # We fit the best fit for the disk contribution to the data
        def residuals(res_disk, i): # Add a constant disk contribution
            mag = -2.5*np.log10((pred_flux[i]+res_disk) * self.star._Proj(self.star.separation) / self.atmo_grid[i].flux0)
            return ((mag + self.atmo_grid[i].ext*par[8] + par[7]) - self.data['mag'][i]) / self.data['mag_err'][i]
        
        def residuals_special(res_disk, i): # Add a constant disk contribution that varies linearly as a function of orbital phase
            mag = -2.5*np.log10((pred_flux[i]+res_disk[0]+res_disk[1]*self.data['phase'][i]) * self.star._Proj(self.star.separation) / self.atmo_grid[i].flux0)
            return ((mag + self.atmo_grid[i].ext*par[8] + par[7]) - self.data['mag'][i]) / self.data['mag_err'][i]
        
        if len(par) > 10:
            disk = np.array(par[9:])
            disk_slope = np.zeros_like(disk)
        else:
            disk = np.ones(self.ndataset) * par[9]
            disk_slope = np.zeros_like(disk)
        chi2 = np.empty(self.ndataset)
        
        # In the case of disk offset fitting
        if offset_free:
            for i in np.arange(self.ndataset):
                # The following is a hack that allows to fit the disk flux with a linear increase
                # as a function of orbital phase. Especially handy for Aug 7 data.
                # The flag for that special fit is the data id containing "_".
                if self.data['id'][i].find('_') != -1:
                    tmp, cov, infodict, mesg, ier = scipy.optimize.leastsq(residuals_special, [disk[i],0.], args=(i,), full_output=1)
                    disk[i] = tmp[0]
                    disk_slope[i] = tmp[1]
                else:
                    tmp, cov, infodict, mesg, ier = scipy.optimize.leastsq(residuals, (disk[i],), args=(i,), full_output=1)
                    disk[i] = np.atleast_1d(tmp)[0]
                chi2[i] = (infodict['fvec']**2).sum()
                par[9+i] = disk[i]
        else:
            for i in np.arange(self.ndataset):
                chi2[i] = (residuals(disk[i], i)**2).sum()
        
        if doKeff:
            pred_Keff = self.Get_Keff(par, nphases=20, make_surface=False, verbose=False)
            chi2Keff = ((self.Keff-pred_Keff)*self.Kefferr)**2
        else:
            pred_Keff = 0.
            chi2Keff = 0.
        
        chi2DM = ((self.DM-par[7])*self.DMerr)**2
        chi2AV = ((self.AV-par[8])*self.AVerr)**2
        
        if verbose:
            disk_str = ""
            for d in disk:
                disk_str += str(d) + ", "
            disk_str = disk_str[:-2]
            for i in xrange(self.ndataset):
                print( "chi2 (%i): %f,   d.o.f.: %i,   avg. companion flux: %.4e,   comp. flux/tot. flux: %.4f,   max. companion flux: %.4e,   max. comp. flux/tot. flux: %.4f,   avg. error: %.4f" %(i, chi2[i], self.data['mag'][i].size, pred_flux[i].mean(), pred_flux[i].mean()/(pred_flux[i].mean()+disk[i]), pred_flux[i].max(), pred_flux[i].max()/(pred_flux[i].max()+disk[i]), self.data['mag_err'][i].mean()) )
            print( "chi2: " + str(chi2.sum()) + ", chi2DM: " + str(chi2DM) + ", chi2AV: " + str(chi2AV) + ", chi2Keff: " + str(chi2Keff) + "\n    Keff: " + str(pred_Keff) + ", disk: " + disk_str )
        
        if return_residuals:
            return np.sqrt(np.r_[chi2, chi2DM, chi2AV, chi2Keff])
        else:
            if return_disk is True:
                return chi2.sum() + chi2DM + chi2AV + chi2Keff, disk, disk_slope
            else:
                return chi2.sum() + chi2DM + chi2AV + chi2Keff

    def Get_flux(self, par, flat=False, func_par=None, make_surface=True, verbose=False):
        """Get_flux(par, flat=False, func_par=None, make_surface=True, verbose=False)
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
            [7]: Distance modulus (not used).
            [8]: Absorption A_V (not used).
            [9-?]: Disk flux.
            Note: If there extra parameters after [9], they are assumed to
            be the individual disk fluxes of each data set.
        flat (False): If True, the values are returned in a 1D vector.
            If False, predicted values are grouped by data set left in a list.
        func_par (None): Function that takes the parameter vector and
            returns the parameter vector. This allow for possible constraints
            on the parameters. The vector returned by func_par must have a length
            equal to the number of expected parameters.
        make_surface (True): If false, Make_surface will not be call
        
        Note: tirr = (par[6]**4 - par[3]**4)**0.25
        
        >>> self.Get_flux([PIBYTWO,1.,0.9,4000.,0.08,1.4,0.07,10.,0.])
        """
        # Apply a function that can modify the value of parameters.
        if func_par is not None:
            par = func_par(par)
        # Verify parameter values to make sure they make sense.
        if par[6] < par[3]: par[6] = par[3]
        # Let's move on with the flux calculation.
        q = par[5] * self.K_to_q
        tirr = (par[6]**4 - par[3]**4)**0.25
        if verbose:
            print( "#####\n" + str(par[0]) + ", " + str(par[1]) + ", " + str(par[2]) + ", " + str(par[3]) + ", " + str(par[4]) + ", " + str(par[5]) + ", " + str(par[6]) + ", " + str(par[7]) + ", " + str(par[8]) + ", " + str(par[9]) + "\n" + "q: " + str(q) + ", tirr: " + str(tirr)  )
        self.star.Make_surface(q=q, omega=par[1], filling=par[2], temp=par[3], tempgrav=par[4], tirr=tirr, porb=self.porb, k1=par[5], incl=par[0])
        flux = []
        if len(par) > 10: # in that situation, individual disk flux values are provided
            for i in np.arange(self.ndataset):
                #print 'Dataset '+str(i)
                flux.append(np.array([self.star.Mag_flux_disk(phase, atmo_grid=self.atmo_grid[i], disk=par[9+i]) for phase in self.data['phase'][i]]))
        elif len(par) == 10: # in that situation, only one disk flux value is provided
            for i in np.arange(self.ndataset):
                #print 'Dataset '+str(i)
                flux.append(np.array([self.star.Mag_flux_disk(phase, atmo_grid=self.atmo_grid[i], disk=par[9]) for phase in self.data['phase'][i]]))
        else: # otherwise there is no disk
            for i in np.arange(self.ndataset):
                #print 'Dataset '+str(i)
                flux.append(np.array([self.star.Mag_flux_disk(phase, atmo_grid=self.atmo_grid[i], disk=0.) for phase in self.data['phase'][i]]))
        if flat:
            return np.hstack(flux)
        else:
            return flux

    def Get_flux_theoretical(self, par, phases, func_par=None, disk_slope=None, return_disk=False):
        """Get_flux_theoretical(par, phases, func_par=None, disk_slope=None, return_disk=False)
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
            [7]: Distance modulus.
            [8]: Absorption A_V.
            [9-?]: Disk flux.
            Note: If there extra parameters after [9], they are assumed to
            be the individual disk fluxes of each data set.
        phases: A list of orbital phases at which the model should be
            evaluated. The list must have the same length as the
            number of data sets, each element can contain many phases.
        func_par (None): Function that takes the parameter vector and
            returns the parameter vector. This allow for possible constraints
            on the parameters. The vector returned by func_par must have a length
            equal to the number of expected parameters.
        disk_slope (None): The disk slope values if provided.
        return_disk (False): If true, will get the disk flux values from
            Calc_chi2_disk.
        
        Note: tirr = (par[6]**4 - par[3]**4)**0.25
        
        >>> self.Get_flux_theoretical([PIBYTWO,1.,0.9,4000.,0.08,1.4,0.07,10.,0.], [[0.,0.25,0.5,0.75]]*4)
        """
        # Apply a function that can modify the value of parameters.
        if func_par is not None:
            par = func_par(par)
        q = par[5] * self.K_to_q
        tirr = (par[6]**4 - par[3]**4)**0.25
        if return_disk:
            chi2, disk, disk_slope = self.Calc_chi2_disk(par, offset_free=1, verbose=False, return_disk=return_disk)
        else:
            self.star.Make_surface(q=q, omega=par[1], filling=par[2], temp=par[3], tempgrav=par[4], tirr=tirr, porb=self.porb, k1=par[5], incl=par[0])
        flux = []
        if len(par) > 10: # in that situation, individual disk flux values are provided
            for i in np.arange(self.ndataset):
                if disk_slope is not None:
                    disk = par[9+i] + disk_slope[i]*phases[i]
                else:
                    disk = np.ones(len(phases[i]))*par[9+i]
                flux.append(np.array([self.star.Mag_flux_disk(phases[i][n], atmo_grid=self.atmo_grid[i], disk=disk[n]) for n in xrange(len(phases[i]))]) + self.atmo_grid[i].ext*par[8] + par[7])
        else:
            for i in np.arange(self.ndataset):
                if disk_slope is not None:
                    disk = par[9] + disk_slope*phases[i]
                else:
                    disk = np.ones(len(phases[i]))*par[9]
                flux.append(np.array([self.star.Mag_flux_disk(phases[i][n], atmo_grid=self.atmo_grid[i], disk=disk[n]) for n in xrange(len(phases[i]))]) + self.atmo_grid[i].ext*par[8] + par[7])
        return flux

    def Get_Keff(self, par, nphases=20, dataset=0, func_par=None, make_surface=False, verbose=False):
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
            [9-?]: Disk flux.
            Note: If there extra parameters after [9], they are assumed to
            be the individual disk fluxes of each data set.
        nphases (20): Number of phases to evaluate the velocity at.
        dataset (int): The index of the atmosphere grid to use for the velocity
            calculation. By default the first one is chosen.
        func_par (None): Function that takes the parameter vector and
            returns the parameter vector. This allow for possible constraints
            on the parameters. The vector returned by func_par must have a length
            equal to the number of expected parameters.
        make_surface (False): Whether lightcurve.make_surface should be called
            or not. If the flux has been evaluate before and the parameters have
            not changed, False is fine.
        verbose (False): Verbosity. Will plot the velocities and the sin fit.
        """
        # Apply a function that can modify the value of parameters.
        if func_par is not None:
            par = func_par(par)
        # If it is required to recalculate the stellar surface.
        if make_surface:
            q = par[5] * self.K_to_q
            tirr = (par[6]**4 - par[3]**4)**0.25
            self.star.Make_surface(q=q, omega=par[1], filling=par[2], temp=par[3], tempgrav=par[4], tirr=tirr, porb=self.porb, k1=par[5], incl=par[0])
        # Deciding which atmosphere grid we use to evaluate Keff
        atmo_grid = self.atmo_grid[dataset]
        # Get the Keffs and fluxes
        phases = np.arange(nphases)/float(nphases)
        Keffs = np.array( [self.star.Flux_disk_Keff(phase, atmo_grid=atmo_grid, disk=0.) for phase in phases] )[:,1]
        tmp = Utils.Misc.Fit_linear(-Keffs, np.sin(cts.TWOPI*(phases)), inline=True)
        if verbose:
            plotxy(-tmp[1]*np.sin(np.linspace(0.,1.)*cts.TWOPI)+tmp[0], np.linspace(0.,1.))
            plotxy(Keffs, phases, line=None, symbol=2)
        Keff = tmp[1]
        return Keff

    def Plot(self, par, nphases=31, verbose=True, device='/XWIN', return_disk=True, func_par=None):
        """Plot(par, nphases=31, verbose=True, device='/XWIN', return_disk=True, func_par=None)
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
            [7]: Distance modulus.
            [8]: Absorption A_V.
            [9-?]: Disk flux.
            Note: If there extra parameters after [9], they are assumed to
            be the individual disk fluxes of each data set.
        nphases (31): Orbital phase resolution of the model
            light curve.
        verbose (True): verbosity.
        device ('/XWIN'): Device driver for Pgplot (can be '/XWIN',
            'filename.ps/PS', 'filename.ps./CPS', '/AQT' (on mac only)).
        return_disk (True): If true, will get the disk flux values from
            Calc_chi2_disk.
        func_par (None): Function that takes the parameter vector and
            returns the parameter vector. This allow for possible constraints
            on the parameters. The vector returned by func_par must have a length
            equal to the number of expected parameters.
        
        >>> self.Plot([PIBYTWO,1.,0.9,4000.,0.08,1.4,0.07,10.,0.])
        """
        if func_par is not None:
            par = func_par(par)
        par = np.asarray(par)
        # Calculate the orbital phases at which the flux will be evaluated
        phases = np.resize(np.linspace(0.,1.,nphases), (self.ndataset, nphases))
        # Fit the data in order to get the offset
        if return_disk:
            chi2, disk, disk_slope = self.Calc_chi2_disk(par, offset_free=1, verbose=verbose, return_disk=return_disk)
        else:
            chi2 = self.Calc_chi2_disk(par.copy(), offset_free=1, verbose=verbose)
            disk_slope = None
        # Calculate the theoretical flux at the orbital phases.
        pred_flux = self.Get_flux_theoretical(par, phases, disk_slope=disk_slope)
        # Loop over the data set and plot the flux, theoretical flux and offset theoretical flux
        rangey = [self.mag.max()+0.3*(self.mag.max()-self.mag.min()), self.mag.min()-0.3*(self.mag.max()-self.mag.min())]
        for i in np.arange(self.ndataset):
            plotxy(self.data['mag'][i], self.data['phase'][i], erry=self.data['mag_err'][i], line=None, symbol=i+2, color=1+i, rangey=rangey, rangex=[0.,1.], labx='Orbital Phase', laby='Magnitude', device=device)
            plotxy(pred_flux[i], phases[i], color=1+i, line=1)
        plotxy([0],[0], color=1)
        y0 = (self.mag.max()+self.mag.min())/2
        dy = (rangey[0]-rangey[1])/35
        # Displaying information about the parameters
        ppgplot.pgsch(0.7) # Make the font smaller
        ppgplot.pgtext(0.4, y0+0*dy, 'Incl.: %4.2f deg'%(par[0]*cts.RADTODEG))
        ppgplot.pgtext(0.4, y0+1*dy, 'Co-rot.: %3.1f'%par[1])
        ppgplot.pgtext(0.4, y0+2*dy, 'Fill.: %5.3f'%par[2])
        ppgplot.pgtext(0.4, y0+3*dy, 'Grav.: %4.2f'%par[4])
        ppgplot.pgtext(0.4, y0+4*dy, 'Temp. back: %7.2f K'%par[3])
        ppgplot.pgtext(0.4, y0+5*dy, 'Temp. front: %7.2f K'%par[6])
        # in the following, we divide the speed by 1000 to convert from m/s to km/s
        ppgplot.pgtext(0.4, y0+6*dy, 'K: %5.2f km/s'%(par[5]/1000))
        ppgplot.pgtext(0.4, y0+7*dy, 'D.M.: %4.2f'%par[7])
        ppgplot.pgtext(0.4, y0+8*dy, 'Aj: %4.2f'%par[8])
        ndisk = len(par)-9
        for i in np.arange(0,ndisk):
            ppgplot.pgtext(0.4, y0+(9+i)*dy, 'Disk %d: %5.3e'%(i+1,par[9+i]))
        ppgplot.pgtext(0.4, y0+(9.5+ndisk)*dy, 'q: %5.3f'%self.star.q)
        ppgplot.pgtext(0.4, y0+(11+ndisk)*dy, 'Chi2: %7.2f, d.o.f.: %i'%(chi2,self.mag.size-len(par)))
        ppgplot.pgsch(1.0) # Restore the font size
        return

    def Prep_plot(self, par, nphases=31, return_disk=True):
        """Prep_plot(par, nphases=31, return_disk=True)
        Returns the lightcurve and orbital phase that would be used to plot the lightcurve.
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
            [9-?]: Disk flux.
            Note: If there extra parameters after [9], they are assumed to
            be the individual disk fluxes of each data set.
        nphases (31): Orbital phase resolution of the model
            light curve.
        return_disk (True): If true, will get the disk flux values from
            Calc_chi2_disk.

        >>> phases, fluxes = Prep_plot(par, nphases)
        phases, fluxes (array): shape (ndataset, nphases)
        """
        phs = np.tile(np.linspace(0, 1, nphases), (self.ndataset,1))
        fluxes = self.Get_flux_theoretical(par, phs, return_disk=return_disk)
        return phs, fluxes

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
            [7]: Distance modulus.
            [8]: Absorption A_V.
            [9]: Disk flux.
            Note: If there extra parameters after [9], they are assumed to
            be the individual disk fluxes of each data set.
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
        DM = par[7]
        A_V = par[8]
        #disk = par[9]
        ndisk = len(par)-9
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
        Lirr_comp = Lirr * self.star.Radius()**2
        eff = Lirr/self.edot
        # we convert Lirr in Lsun units
        Lirr /= 3.839e33
        if verbose:
            print( "##### Pretty Print #####" )
            print( "%9.7f, %3.1f, %9.7f, %10.5f, %4.2f, %9.7f, %9.7f, %6.3f, %6.3f, %5.4e" %tuple(par[:10]) )
            print( "" )
            print( "Corotation factor: %4.2f" %corot )
            print( "Gravity Darkening: %5.3f" %gdark )
            print( "" )
            print( "Filling factor: %6.4f" %fill )
            print( "Orbital separation: %5.4e km" %(separation/1000) )
            print( "Roche lobe size: %6.4f (orb. sep.)" %roche )
            print( "" )
            print( "Irradiation efficiency: %6.4f" %eff )
            print( "Irradiation luminosity (at the pulsar): %5.4e Lsun" %Lirr )
            print( "Irradiation luminosity (at the companion's surface): %5.4e erg/s" %Lirr_comp )
            print( "Backside temperature: %7.2f K" %temp_back )
            print( "Frontside temperature: %7.2f (tabul.), %7.2f (approx.) K" %(np.exp(self.star.logteff.max()),temp_front) )
            print( "" )
            print( "Distance Modulus: %6.3f" %DM )
            print( "Absorption (V band): %6.3f" %A_V )
            print( "" )
            for i in np.arange(0,ndisk):
                print( "Disk flux %d: %6.4e" %(i+1,par[9+i]) )
            print( "" )
            print( "Inclination: %5.3f rad (%6.2f deg)" %(incl,incl*cts.RADTODEG) )
            print( "K: %7.3f km/s" %(K/1000) )
            print( "" )
            print( "Mass ratio: %6.3f" %q )
            print( "Mass NS: %5.3f Msun" %Mns )
            print( "Mass Comp: %5.3f Msun" %Mwd )
        return np.r_[corot,gdark,fill,separation,roche,eff,tirr,temp_back,np.exp(self.star.logteff.max()),temp_front,DM,A_V,incl,incl*cts.RADTODEG,K,q,Mns,Mwd]

    def __Read_atmo(self, atmo_fln):
        """__Read_atmo(atmo_fln)
        Reads the atmosphere model data.
        
        atmo_fln: A file containing the grid model information for each
            data set. The format of each line of the file is as follows:
                band_name, center_wavelength, delta_wavelength, flux0,
                    extinction, grid_file
        
        Note: if a line starts with #!, the atmosphere model will be used
            to calculate the effective velocity and other quantities.
        
        >>> self.__Read_atmo(atmo_fln)
        """
        f = open(atmo_fln,'r')
        lines = f.readlines()
        self.atmo_grid = []
        for line in lines:
            if (line[0] != '#') and (line[0] != '\n'):
                tmp = line.split()
                # We read the BT-Settl.7 data with the proper class
                if tmp[5].find('BT-Settl.7') != -1:
                    self.atmo_grid.append( Atmosphere.Atmo_grid_BTSettl7(tmp[5], float(tmp[1]), float(tmp[2]), float(tmp[3]), float(tmp[4]), logg_lims=[3.0,4.5]) )
                else:
                    self.atmo_grid.append(Atmosphere.Atmo_grid(tmp[5], float(tmp[1]), float(tmp[2]), float(tmp[3]), float(tmp[4])))
        return

    def __Read_data(self, data_fln):
        """__Read_data(self, data_fln)
        Reads the photometric data.
        
        data_fln: A file containing the information for each data set.
            The format of the file is as follows:
                band_name, column_phase, column_flux, column_error_flux,
                    shift_to_phase_zero, calibration_error, data_file
        
        >>> self.__Read_data(data_fln)
        """
        f = open(data_fln,'r')
        lines = f.readlines()
        self.data = {'mag':[], 'phase':[], 'mag_err':[], 'calib':[], 'fln':[], 'id':[]}
        for line in lines:
            if (line[0] != '#') and (line[0] != '\n'):
                tmp = line.split()
                d = np.loadtxt(tmp[6], usecols=[int(tmp[1]),int(tmp[2]),int(tmp[3])], unpack=True)
                # With the flag '_' in the observation id, we do not take %1 so that
                # we preserve the long-term phase coherence.
                if tmp[0].find('_') != -1:
                    self.data['phase'].append((d[0] - float(tmp[4])))
                else:
                    self.data['phase'].append((d[0] - float(tmp[4])) % 1.)
                self.data['mag'].append(d[1])
                self.data['mag_err'].append(d[2])
                self.data['calib'].append(float(tmp[5]))
                self.data['fln'].append(tmp[6])
                self.data['id'].append(tmp[0])
        return
            
    def __Setup(self):
        """__Setup()
        Stores some important information in class variables.
        
        >>> self.__Setup()
        """
        # We calculate the constant for the conversion of K to q (observed
        # velocity semi-amplitude to mass ratio, with K in m/s)
        self.K_to_q = Utils.Binary.Get_K_to_q(self.porb, self.x2sini)
        # Storing values in 1D arrays.
        ext = []
        self.data['ext'] = []
        for i in np.arange(self.ndataset):
            ext.extend(self.data['phase'][i]*0.+self.atmo_grid[i].ext)
            self.data['ext'].append(self.atmo_grid[i].ext)
        self.ext = np.asarray(ext)
        self.data['ext'] = np.asarray(self.data['ext'])
        self.data['calib'] = np.asarray(self.data['calib'])
        self.mag = np.hstack(self.data['mag'])
        self.phase = np.hstack(self.data['phase'])
        self.mag_err = np.hstack(self.data['mag_err'])
        return

######################## class Photometry_disk ########################

