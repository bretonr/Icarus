# Licensed under a 3-clause BSD style license - see LICENSE

__all__ = ["Photometry_disk"]

from ..Utils.import_modules import *
from .. import Utils
from .. import Core
from .. import Atmosphere


######################## class Photometry_disk ########################
class Photometry_disk:
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
    def __init__(self, atmo_fln, data_fln, nalf, porb, x2sini, edot=1., DM=0., DMerr=0., AJ=0., AJerr=0., Keff=0., Kefferr=0., read=False):
        """__init__(atmo_fln, data_fln, nalf, porb, x2sini, edot=1., DM=0., DMerr=0., AJ=0., AJerr=0., Keff=0., Kefferr=0., read=False)
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
        nalf: The number of surface slice. Defines how coarse/fine the
            surface grid is.
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
        AJ (0.): J band absorption to the source, if known.
        AJerr (0.): Reciprocal of the J band absorption error to the source,
            if known. A value of 0. will disable the chi2AJ contribution.
        Keff (0.): Effective projected velocity semi-amplitude in m/s, if known.
        Kefferr (0.): Reciprocal of the effective projected velocity
            semi-amplitude in m/s, if known.
        read (False): Whether the geodesic surface should be read from a file or
            generated from scratch.
        
        >>> fit = Photometry(atmo_fln, data_fln, nalf, porb, x2sini, edot)
        """
        # We define some class attributes.
        self.porb = porb
        self.x2sini = x2sini
        self.edot = edot
        self.DM = DM
        self.DMerr = DMerr
        self.AJ = AJ
        self.AJerr = AJerr
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
        self.lightcurve = Core.Star_disk(nalf, read=read)
        self.__Setup()

    def Calc_chi2(self, par, func_par=None, offset_free=1, verbose=False):
        """Calc_chi2(par, func_par=None, offset_free=1, verbose=False)
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
            [8]: Absorption A_J (can be None).
            [9-?]: Disk flux.
            Note: If there extra parameters after [9], they are assumed to
            be the individual disk fluxes of each data set.
            Note: DM and A_J can be set to None. In which case, if
            offset_free = 1, these parameters will be fit for.
        func_par (None): Function that takes the parameter vector and
            returns the parameter vector. This allow for possible constraints
            on the parameters. The vector returned by func_par must have a length
            equal to the number of expected parameters.
        offset_free (1):
            1) offset_free = 1:
            A "post" fit is performed in order to adjust the offsets of the
            curves accounting for the fact that the absolute calibration of
            the photometry may vary.
            Note:
            The errors should be err**2 = calib_err**2 + 1/sum(flux_err)**2
            but we neglect the second term because it is negligeable.
            2) offset_free = 0:
            If the offset is not free and the DM and A_J are known up to some
            uncertainty, the calculation below is as following. Refer to
            fitcurve.for for more information. Should be:
                chi2 = sum[ w_i*(off_i-dm-aj*C_i)**2]
                    + w_dm*(dm-dm_obs)**2 
                    + w_aj*(aj-aj_obs)**2,     with w = 1/sigma**2
            3) offset_free = 2:
            Like for offset_free = 1, but the offset for each photometric
            band is also returned as well as the full parameter vector (after
            applying the corrections and the linear fit for DM and A_J. This
            is used, for instance, by self.Plot(par).
            4) offset_free = 3:
            Like for offset_free = 1, but the residuals are returned.
            (chi2 = sum(residuals**2))
        verbose (False): If true will display the list of parameters.
        
        >>> self.Calc_chi2([PIBYTWO,1.,0.9,4000.,0.08,1.4,0.07,10.,0.])
        """
        # We can provide a function that massages the input parameters and returns them.
        # This function can, for example, handle fixed parameters or boundary limits.
        if func_par is not None:
            par = func_par(par)
        if offset_free:
            # Calculate the theoretical flux
            pred_flux = self.Get_flux(par, flat=False, verbose=verbose)
            # Calculate the residuals between observed and theoretical flux
            res1 = numpy.array([ Utils.fit_linear(self.data['mag'][i]-pred_flux[i], err=self.data['err'][i], m=0., inline=True) for i in numpy.arange(self.ndataset) ])
            # Fit for the best offset between the observed and theoretical flux given the DM and A_J
            res2 = Utils.fit_linear(res1[:,0], self.data['ext'], err=self.data['calib'], b=par[7], m=par[8], inline=True)
            par[7], par[8] = res2[0], res2[1]
            chi2 = res1[:,2].sum() + res2[2]
            if verbose:
                print( 'chi2(1): '+str(res1[:,2].sum())+', chi2(2): '+str(res2[2])+', chi2: '+str(chi2)+', D.M. : '+str(par[7])+', Aj: '+str(par[8]) )
            if offset_free == 2:
                return chi2, res1[:,0]-(self.data['ext']*par[8] + par[7]), par
            elif offset_free == 3:
                return numpy.sqrt(numpy.r_[res1[:,2], res2[2]])
            else:
                return chi2
        else:
            pred_flux = self.Get_flux(par, flat=True)
            ((par[7],par[8]), chi2, rank, s) = Utils.fit_linear(self.mag-pred_flux, x=self.ext, err=self.err, b=par[7], m=par[8])
            chi2DM = ((self.DM-par[7])*self.DMerr)**2
            chi2AJ = ((self.AJ-par[8])*self.AJerr)**2
            return chi2 + chi2DM + chi2AJ

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
            [8]: Absorption A_J (can be None).
            [9-?]: Disk flux.
            Note: If there extra parameters after [9], they are assumed to
            be the individual disk fluxes of each data set.
            Note: Unlike Calc_chi2, DM and A_J cannot be fitted for.
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
        self.lightcurve.Make_surface(q=q, omega=par[1], filling=par[2], temp=par[3], tempgrav=par[4], tirr=tirr, porb=self.porb, k1=par[5], incl=par[0])
        pred_flux = [numpy.array([self.lightcurve.Flux_disk(phase, atmo_grid=self.atmo_grid[i], disk=0.) for phase in self.data['phase'][i]]) for i in numpy.arange(self.ndataset)]
        
        # We fit the best fit for the disk contribution to the data
        def residuals(res_disk, i): # Add a constant disk contribution
            mag = -2.5*numpy.log10((pred_flux[i]+res_disk) * self.lightcurve._Proj(self.lightcurve.separation) / self.atmo_grid[i].flux0)
            return ((mag + self.atmo_grid[i].ext*par[8] + par[7]) - self.data['mag'][i]) / self.data['err'][i]
        
        def residuals_special(res_disk, i): # Add a constant disk contribution that varies linearly as a function of orbital phase
            mag = -2.5*numpy.log10((pred_flux[i]+res_disk[0]+res_disk[1]*self.data['phase'][i]) * self.lightcurve._Proj(self.lightcurve.separation) / self.atmo_grid[i].flux0)
            return ((mag + self.atmo_grid[i].ext*par[8] + par[7]) - self.data['mag'][i]) / self.data['err'][i]
        
        if len(par) > 10:
            disk = par[9:]
            disk_slope = numpy.zeros_like(disk)
        else:
            disk = numpy.ones(self.ndataset) * par[9]
            disk_slope = numpy.zeros_like(disk)
        chi2 = numpy.empty(self.ndataset)
        
        # In the case of disk offset fitting
        if offset_free:
            for i in numpy.arange(self.ndataset):
                # The following is a hack that allows to fit the disk flux with a linear increase
                # as a function of orbital phase. Especially handy for Aug 7 data.
                # The flag for that special fit is the data id containing "_".
                if self.data['id'][i].find('_') != -1:
                    tmp, cov, infodict, mesg, ier = scipy.optimize.leastsq(residuals_special, [disk[i],0.], args=(i,), full_output=1)
                    disk[i] = tmp[0]
                    disk_slope[i] = tmp[1]
                else:
                    tmp, cov, infodict, mesg, ier = scipy.optimize.leastsq(residuals, (disk[i],), args=(i,), full_output=1)
                    disk[i] = numpy.atleast_1d(tmp)[0]
                chi2[i] = (infodict['fvec']**2).sum()
                par[9+i] = disk[i]
        else:
            for i in numpy.arange(self.ndataset):
                chi2[i] = (residuals(disk[i], i)**2).sum()
        
        if doKeff:
            pred_Keff = self.Get_Keff(par, nphases=20, make_surface=False, verbose=False)
            chi2Keff = ((self.Keff-pred_Keff)*self.Kefferr)**2
        else:
            pred_Keff = 0.
            chi2Keff = 0.
        
        chi2DM = ((self.DM-par[7])*self.DMerr)**2
        chi2AJ = ((self.AJ-par[8])*self.AJerr)**2
        
        if verbose:
            disk_str = ""
            for d in disk:
                disk_str += str(d) + ", "
            disk_str = disk_str[:-2]
            for i in xrange(self.ndataset):
                print( "chi2 (%i): %f,   d.o.f.: %i,   avg. companion flux: %.4e,   comp. flux/tot. flux: %.4f, avg. error: %.4f" %(i,chi2[i],self.data['mag'][i].size, pred_flux[i].mean(), pred_flux[i].mean()/(pred_flux[i].mean()+disk[i]), self.data['err'][i].mean()) )
            print( "chi2: " + str(chi2.sum()) + ", chi2DM: " + str(chi2DM) + ", chi2AJ: " + str(chi2AJ) + ", chi2Keff: " + str(chi2Keff) + "\n    Keff: " + str(pred_Keff) + ", disk: " + disk_str )
        
        if return_residuals:
            return numpy.sqrt(numpy.r_[chi2, chi2DM, chi2AJ, chi2Keff])
        else:
            if return_disk is True:
                return chi2.sum() + chi2DM + chi2AJ + chi2Keff, disk, disk_slope
            else:
                return chi2.sum() + chi2DM + chi2AJ + chi2Keff

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
            [8]: Absorption A_J (not used).
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
        self.lightcurve.Make_surface(q=q, omega=par[1], filling=par[2], temp=par[3], tempgrav=par[4], tirr=tirr, porb=self.porb, k1=par[5], incl=par[0])
        flux = []
        if len(par) > 10: # in that situation, individual disk flux values are provided
            for i in numpy.arange(self.ndataset):
                #print 'Dataset '+str(i)
                flux.append(numpy.array([self.lightcurve.Mag_flux_disk(phase, atmo_grid=self.atmo_grid[i], disk=par[9+i]) for phase in self.data['phase'][i]]))
        else:
            for i in numpy.arange(self.ndataset):
                #print 'Dataset '+str(i)
                flux.append(numpy.array([self.lightcurve.Mag_flux_disk(phase, atmo_grid=self.atmo_grid[i], disk=par[9]) for phase in self.data['phase'][i]]))
        if flat:
            return numpy.hstack(flux)
        else:
            return flux

    def Get_flux_theoretical(self, par, phases, func_par=None, disk_slope=None):
        """Get_flux_theoretical(par, phases, func_par=None, disk_slope=None)
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
            [8]: Absorption A_J.
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
        
        Note: tirr = (par[6]**4 - par[3]**4)**0.25
        
        >>> self.Get_flux_theoretical([PIBYTWO,1.,0.9,4000.,0.08,1.4,0.07,10.,0.], [[0.,0.25,0.5,0.75]]*4)
        """
        # Apply a function that can modify the value of parameters.
        if func_par is not None:
            par = func_par(par)
        q = par[5] * self.K_to_q
        tirr = (par[6]**4 - par[3]**4)**0.25
        self.lightcurve.Make_surface(q=q, omega=par[1], filling=par[2], temp=par[3], tempgrav=par[4], tirr=tirr, porb=self.porb, k1=par[5], incl=par[0])
        flux = []
        if len(par) > 10: # in that situation, individual disk flux values are provided
            for i in numpy.arange(self.ndataset):
                if disk_slope is not None:
                    disk = par[9+i] + disk_slope[i]*phases[i]
                else:
                    disk = numpy.ones(len(phases[i]))*par[9+i]
                flux.append(numpy.array([self.lightcurve.Mag_flux_disk(phases[i][n], atmo_grid=self.atmo_grid[i], disk=disk[n]) for n in xrange(len(phases[i]))]) + self.atmo_grid[i].ext*par[8] + par[7])
        else:
            for i in numpy.arange(self.ndataset):
                if disk_slope is not None:
                    disk = par[9] + disk_slope*phases[i]
                else:
                    disk = numpy.ones(len(phases[i]))*par[9]
                flux.append(numpy.array([self.lightcurve.Mag_flux_disk(phases[i][n], atmo_grid=self.atmo_grid[i], disk=disk[n]) for n in xrange(len(phases[i]))]) + self.atmo_grid[i].ext*par[8] + par[7])
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
            [9-?]: Disk flux.
            Note: If there extra parameters after [9], they are assumed to
            be the individual disk fluxes of each data set.
        nphases (20): Number of phases to evaluate the velocity at.
        dataset (None): The dataset for which the velocity is evaluated
            (i.e. the atmosphere grid to use).
            This parameter must be set if not atmosphere grid was specified
            for the Keff evaluation in the class initialization.
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
        Keffs = numpy.array( [self.lightcurve.Flux_disk_Keff(phase, atmo_grid=atmo_grid, disk=0.) for phase in phases] )[:,1]
        tmp = Utils.fit_linear(-Keffs, numpy.sin(TWOPI*(phases)), inline=True)
        if verbose:
            plotxy(-tmp[1]*numpy.sin(numpy.linspace(0.,1.)*TWOPI)+tmp[0], numpy.linspace(0.,1.))
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
            [8]: Absorption A_J.
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
        par = numpy.array(par)
        # Calculate the orbital phases at which the flux will be evaluated
        phases = numpy.resize(numpy.linspace(0.,1.,nphases), (self.ndataset, nphases))
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
        for i in numpy.arange(self.ndataset):
            plotxy(self.data['mag'][i], self.data['phase'][i], erry=self.data['err'][i], line=None, symbol=i+2, color=1+i, rangey=rangey, rangex=[0.,1.], labx='Orbital Phase', laby='Magnitude', device=device)
            plotxy(pred_flux[i], phases[i], color=1+i, line=1)
        plotxy([0],[0], color=1)
        y0 = (self.mag.max()+self.mag.min())/2
        dy = (rangey[0]-rangey[1])/35
        # Displaying information about the parameters
        ppgplot.pgsch(0.7) # Make the font smaller
        ppgplot.pgtext(0.4, y0+0*dy, 'Incl.: %4.2f deg'%(par[0]*RADTODEG))
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
        for i in numpy.arange(0,ndisk):
            ppgplot.pgtext(0.4, y0+(9+i)*dy, 'Disk %d: %5.3e'%(i+1,par[9+i]))
        ppgplot.pgtext(0.4, y0+(9.5+ndisk)*dy, 'q: %5.3f'%self.lightcurve.q)
        ppgplot.pgtext(0.4, y0+(11+ndisk)*dy, 'Chi2: %7.2f, d.o.f.: %i'%(chi2,self.mag.size-len(par)))
        ppgplot.pgsch(1.0) # Restore the font size
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
            [7]: Distance modulus.
            [8]: Absorption A_J.
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
        A_J = par[8]
        #disk = par[9]
        ndisk = len(par)-9
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
            print( "Irradiation luminosity: %5.4e Lsun" %Lirr )
            print( "Backside temperature: %7.2f K" %temp_back )
            print( "Frontside temperature: %7.2f (tabul.), %7.2f (approx.) K" %(numpy.exp(self.lightcurve.logteff.max()),temp_front) )
            print( "" )
            print( "Distance Modulus: %6.3f" %DM )
            print( "Absorption (J band): %6.3f" %A_J )
            print( "" )
            for i in numpy.arange(0,ndisk):
                print( "Disk flux %d: %6.4e" %(i+1,par[9+i]) )
            print( "" )
            print( "Inclination: %5.3f rad (%6.2f deg)" %(incl,incl*RADTODEG) )
            print( "K: %7.3f km/s" %(K/1000) )
            print( "" )
            print( "Mass ratio: %6.3f" %q )
            print( "Mass NS: %5.3f Msun" %Mns )
            print( "Mass Comp: %5.3f Msun" %Mwd )
        return numpy.r_[corot,gdark,fill,separation,roche,eff,tirr,temp_back,numpy.exp(self.lightcurve.logteff.max()),temp_front,DM,A_J,incl,incl*RADTODEG,K,q,Mns,Mwd]

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
            elif (line[:2] == '#!'):
                tmp = line.split()
                tmp = tmp[1:]
                if tmp[5].find('BT-Settl.7') != -1:
                    self.keff_atmo_grid = Atmosphere.Atmo_grid_BTSettl7(tmp[5], float(tmp[1]), float(tmp[2]), float(tmp[3]), float(tmp[4]))
                else:
                    self.keff_atmo_grid = Atmosphere.Atmo_grid(tmp[5], float(tmp[1]), float(tmp[2]), float(tmp[3]), float(tmp[4]))
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
        self.data = {'mag':[], 'phase':[], 'err':[], 'calib':[], 'fln':[], 'id':[]}
        for line in lines:
            if (line[0] != '#') and (line[0] != '\n'):
                tmp = line.split()
                d = numpy.loadtxt(tmp[6], usecols=[int(tmp[1]),int(tmp[2]),int(tmp[3])], unpack=True)
                # With the flag '_' in the observation id, we do not take %1 so that
                # we preserve the long-term phase coherence.
                if tmp[0].find('_') != -1:
                    self.data['phase'].append((d[0] - float(tmp[4])))
                else:
                    self.data['phase'].append((d[0] - float(tmp[4])) % 1.)
                self.data['mag'].append(d[1])
                self.data['err'].append(d[2])
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
        self.K_to_q = Utils.Get_K_to_q(self.porb, self.x2sini)
        # Storing values in 1D arrays.
        ext = []
        self.data['ext'] = []
        for i in numpy.arange(self.ndataset):
            ext.extend(self.data['phase'][i]*0.+self.atmo_grid[i].ext)
            self.data['ext'].append(self.atmo_grid[i].ext)
        self.ext = numpy.array(ext)
        self.data['ext'] = numpy.array(self.data['ext'])
        self.data['calib'] = numpy.array(self.data['calib'])
        self.mag = numpy.hstack(self.data['mag'])
        self.phase = numpy.hstack(self.data['phase'])
        self.err = numpy.hstack(self.data['err'])
        return

######################## class Photometry_disk ########################
