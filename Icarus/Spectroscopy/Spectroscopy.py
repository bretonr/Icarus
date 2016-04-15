# Licensed under a 3-clause BSD style license - see LICENSE

__all__ = ["Spectroscopy", "Doppler_shift", "Normalize_spectrum", "Rebin", "Process_flux", "Process_flux1"]

import sys
import glob

from ..Utils.import_modules import *
from .. import Utils
from .. import Core
from .. import Atmosphere

logger = logging.getLogger(__name__)


######################## class Spectroscopy ########################
class Spectroscopy(object):
    """Spectroscopy
    This class allows to fit the flux from the primary star
    of a binary system, assuming it is heated by the secondary
    (which in most cases will be a pulsar).
    
    It is meant to deal with spectroscopic data. A set of spectroscopic
    data (i.e. different orbital phases) is read. For each data set, one can
    calculate the predicted flux of the model at every data point (i.e.
    for a given orbital phase).
    """
    def __init__(self, atmo_grid, data_fln, ndiv, porb, x2sini, phase_offset=-0.25, seeing=-1, read=True, oldchi=False):
        """
        This class allows to fit the flux from the primary star
        of a binary system, assuming it is heated by the secondary
        (which in most cases will be a pulsar).
    
        It is meant to deal with spectroscopic data. A set of spectroscopic
        data (i.e. different orbital phases) is read. For each data set, one can
        calculate the predicted flux of the model at every data point (i.e.
        for a given orbital phase).
        
        atmo_grid: An atmosphere grid instance or a file containing the grid
            model information for the whole data set. The format of each line
            of the file is as follow:
                descriptor name
                grid file
                wavelength cut low
                wavelength cut high
                oversample factor
                smoothing factor
            Can also be None. In which case, the atmosphere grid must be given
            for each Get_flux call.
        data_fln: A file containing the information for each data set.
            The format of the file is as follow:
                descriptor name
                orbital phase
                column wavelength
                column flux
                column error flux
                data file
                velocity offset
                velocity offset error
                approximate velocity
        ndiv: The number of surface element subdivisions. Defines how coarse/fine
            the surface grid is.
        porb: Orbital period of the system in seconds.
        x2sini: Projected semi-major axis of the secondary (pulsar)
            in light-second.
        phase_offset (-0.25): Value to be added to the orbital phase in order
            to have phase 0.0 and 0.5 being conjunction times, with 0.0 the eclipse.
        seeing (-1): The seeing factor. -1 will use the default value.
        read (bool): If True, Icarus will use the pre-calculated geodesic
            primitives. This is the recommended option, unless you have the
            pygts package installed to calculate it on the spot.
        
        >>> fit = Spectroscopy(atmo_fln, data_fln, ndiv, porb, x2sini)
        """
        # We define some class attributes.
        self.porb = porb
        self.x2sini = x2sini
        # We read the data.
        print( 'Reading spectral data' )
        self.__Read_data(data_fln, phase_offset=phase_offset)
        # We read the atmosphere models with the atmo_grid class
        print( 'Reading atmosphere grid' )
        if atmo_grid is None:
            self.atmo_grid = None
        elif isinstance(atmo_grid, basestring):
            self.__Read_atmo(atmo_grid)
        else:
            self.atmo_grid = atmo_grid
        # We keep in mind the number of datasets
        self.ndataset = len(self.data['phase'])
        # We initialize some important class attributes.
        print( 'Initializing the lightcurve attribute' )
        self.star = Core.Star(ndiv, atmo_grid=self.atmo_grid, read=read, oldchi=oldchi)
        print( 'Performing some more initialization' )
        self.Initialize(seeing=seeing)
        print( 'Done. Play and have fun...' )

    def Fit_flux(self, flux_model, v=None, inds=None):
        """
        Given a set of flux models calculated with self.Get_flux, performs the
        resampling to match the data and adjustment of the continuum using a polynomial.

        Returns the adjusted fluxes and the individual chi2.

        flux_model (list): Flux models calculated by self.Get_flux. The sampling
            is that of the atmosphere grid.
        v (list): List of velocities to shift the input fluxes in m/s.
            If None, will assume 0 m/s.
        inds (list): List of indices of the data corresponding to the provided
            fluxes. If None, assumes that there is an entry for each data.

        >>> fluxes, chi2 = Fit_flux(flux_model)
        """
        if v is None:
            v = [0.]*self.ndataset
        if inds is None:
            inds = np.arange(self.ndataset)
        fluxes, chi2 = zip(*[ Process_flux(self.data['flux'][i], self.data['err'][i], flux_model[i], self.data['wavelength'][i], self.atmo_grid.wav, z=v[i]/cts.c) for i in inds ])
        return fluxes, chi2

    def Get_flux(self, par, orbph=None, velocities=0., gravscale=None, atmo_grid=None, verbose=False):
        """Get_flux(par, orbph=None, velocities=0., gravscale=None, atmo_grid=None, verbose=False)
        Returns the predicted flux by the model evaluated at the
        observed values in the data set.
        
        par (array): Parameter list.
            [0]: Orbital inclination in radians.
            [1]: Corotation factor.
            [2]: Roche-lobe filling.
            [3]: Companion temperature.
            [4]: Gravity darkening coefficient.
            [5]: K (projected velocity semi-amplitude) in m/s.
            [6]: Front side temperature.
            [7]: Systematic velocity offset in m/s.
            Note: Can also be a dictionary:
                par.keys() = ['incl', 'corotation', 'filling', 'tnight',
                    'gravdark', 'k1', 'tday', 'vsys']
        orbph (float, array): List of orbital phases to evaluate the
            spectrum at. If None, will use the same phaes as the data.
        velocities (float, array): Velocities to add to the
            pre-computed ones (in m/s unit).
        gravscale (optional): gravitational scaling parameter.
        atmo_grid (optional): atmosphere grid instance used to
            calculate the flux.
        verbose (False): If true will display the list of parameters.
        
        Note: tirr = (par[6]**4 - par[3]**4)**0.25
        
        >>> self.Get_flux([PIBYTWO,1.,0.9,4000.,0.08,300e3,6000.,50e3])
        """
        logger.log(9, "start")
        ## Check if we are dealing with a dictionary
        if isinstance(par, dict):
            par = [par['incl'], par['corotation'], par['filling'], par['tnight'], par['gravdark'], par['k1'], par['tday'], par['vsys']]
        if orbph is None:
            orbph = self.data['phase']
        else:
            orbph = np.atleast_1d(orbph)
        if atmo_grid is None:
            atmo_grid = self.atmo_grid
        #velocities = np.zeros_like(orbph) + par[7] + self.data['v_offset'] + velocities
        velocities = np.zeros_like(orbph) + par[7] + velocities
        q = par[5] * self.K_to_q
        tirr = (par[6]**4 - par[3]**4)**0.25

        if verbose:
            print( "#####\n" + str(par[0]) + ", " + str(par[1]) + ", " + str(par[2]) + ", " + str(par[3]) + ", " + str(par[4]) + ", " + str(par[5]) + ", " + str(par[6]) + ", " + str(par[7]) + "\n" + "q: " + str(q) + ", tirr: " + str(tirr)  )

        self.star.Make_surface(q=q, omega=par[1], filling=par[2], temp=par[3], tempgrav=par[4], tirr=tirr, porb=self.porb, k1=par[5], incl=par[0])
        flux = [self.star.Flux_doppler(phs, velocity=velocity, gravscale=gravscale, atmo_grid=atmo_grid) for phs,velocity in zip(orbph,velocities)]
        logger.log(9, "end")
        return flux

    def Initialize(self, seeing=-1):
        """Initialize(seeing=-1)
        Initializes and stores some important variables
        
        seeing (-1): The seeing factor. -1 will use the default value.
        """
        ## We calculate the constant for the conversion of K to q (observed
        ## velocity semi-amplitude to mass ratio, with K in m/s)
        self.K_to_q = Utils.Binary.Get_K_to_q(self.porb, self.x2sini)

        ## Here we pre-calculate the wavelengths, interpolation indices
        ## and weights for the rebining of the log-spaced atmo_grid data
        ## to linear
        if 0:
            self.wavelength = []
            self.binfactor = []
            self.ws_loglin = []
            self.inds_loglin = []
            self.ws_rebin = []
            self.inds_rebin = []
            self.sigma = []
            for i in np.arange(self.ndataset):
                ## Checking if the bounds of the data are within those of the atmosphere grid
                if self.data['wavelength'][i][0] < self.atmo_grid.wav[0]:
                    print( "Warning: the data wavelength coverage for {} is out of the atmosphere grid lower bound.".format(self.data['id'][i]) )
                if self.data['wavelength'][i][-1] > self.atmo_grid.wav[-1]:
                        print( "Warning: the data wavelength coverage for {} is out of the atmosphere grid upper bound.".format(self.data['id'][i]) )
                stepsize = self.data['wavelength'][i][1]-self.data['wavelength'][i][0]
                # binfactor is the oversampling factor of the model vs observed spectrum
                tmp = np.floor(stepsize/(self.atmo_grid.wav[1]-self.atmo_grid.wav[0]))
                if tmp < 1: tmp = 1
                self.binfactor.append( tmp )
                stepsize /= self.binfactor[i]
                self.wavelength.append( np.arange(self.data['wavelength'][i][0],self.data['wavelength'][i][-1]+stepsize/2,stepsize) )
                # To rebin from log to lin
                ws, inds = Utils.Series.Getaxispos_vector(self.atmo_grid.wav, self.wavelength[i]+0.0)
                self.ws_loglin.append(ws)
                self.inds_loglin.append(inds)
                # To rebin to the observed wavelengths
                ws, inds = Utils.Series.Getaxispos_vector(self.wavelength[i], self.data['wavelength'][i]+0.0)
                self.ws_rebin.append(ws)
                self.inds_rebin.append(inds)
        return

    def Plot(self, par=None, flux_model=None, wav_model=None, inds=None, panels=None, plotobs=True, plotmodel=True, plotres=True):
        """
        Plots the observed and predicted values along with the light curve.

        Either a list of parameters can be provided, or a list of model fluxes.

        par (list): Parameter list.
            [0]: Orbital inclination in radians.
            [1]: Corotation factor.
            [2]: Roche-lobe filling.
            [3]: Companion temperature.
            [4]: Gravity darkening coefficient.
            [5]: K (projected velocity semi-amplitude) in m/s.
            [6]: Front side temperature.
            [7]: Systematic velocity offset in m/s.
        flux_model (list): The list of model fluxes.
        wav_model (list): The list of model wavelength.
        inds: Indices of the data to be plotted. If None will plot all of them.
        panels (list): The layout of the plot. Should be a list: (ncol, nrows).
            If none, there will be only one spectrum per window.
        plotobs (bool): If True will plot the observed data.
        plotmodel (bool): If True will plot the model.
        plotres (bool): If True will plot the residuals.

        >>> self.Plot([PIBYTWO,1.,0.9,4000.,0.08,1.4,0.07,10.,0.])
        """
        if inds is None:
            inds = np.arange(self.ndataset)
        if isinstance(inds, int):
            inds = [inds]

        ## Retrieving the observed fluxes, if required
        if plotobs or plotres:
            flux_obs = [self.data['flux'][i] for i in inds]
            flux_obs_err = [self.data['err'][i] for i in inds]
            wav_obs = [self.data['wavelength'][i] for i in inds]

        ## If the model fluxes are not provided, we calculate them
        if plotmodel or plotres:
            if flux_model is None:
                flux_model = self.Get_flux(par, orbph=self.data['phase'][inds], velocities=self.data['v_offset'][inds], verbose=False)
                flux_model, chi2 = self.Fit_flux(flux_model, inds=inds)
            if wav_model is None:
                if len(flux_model[0]) == len(self.data['wavelength'][inds[0]]):
                    wav_model = [self.data['wavelength'][i] for i in inds]
                else:
                    if len(flux_model[0]) == self.atmo_grid.wav.size:
                        wav_model = [self.atmo_grid.wav]*len(inds)
                    else:
                        raise Exception("The model fluxes don't match the data nor the atmosphere grid dimension.")

        ## Connecting to the plotting figure
        if len(pylab.get_fignums()) == 0:
            fig = pylab.figure()
        else:
            fig = pylab.gcf()

        ## Plotting
        if panels is None:
            panels = [1,1]
        llim = 0.05
        blim = 0.05
        wlim = 0.92
        hlim = 0.93
        colspace = 0.03
        rowspace = 0.03
        if plotres and (plotmodel or plotobs):
            split_ratio = 0.3
        else:
            split_ratio = 0.0
        w = (wlim - (panels[0]-1)*colspace)/panels[0]
        h = (hlim - (panels[1]-1)*rowspace)/panels[1]
        for i in xrange(len(inds)):
            if i%(panels[0]*panels[1]) == 0:
                fig.clf()
            col = (i%(panels[0]*panels[1]))/panels[1]
            row = (i%(panels[0]*panels[1]))%panels[1]
            ax1lim = [llim+col*(colspace+w),(blim+(panels[1]-1-row)*(rowspace+h)+split_ratio*h),w,h*(1-split_ratio)]
            ax2lim = [llim+col*(colspace+w),(blim+(panels[1]-1-row)*(rowspace+h)),w,h*split_ratio]
            ax = fig.add_axes(ax1lim)
            if plotobs:
                ax.plot(wav_obs[i], flux_obs[i], 'k-')
                ax.set_xlim([wav_obs[i][0],wav_obs[i][-1]])
            if plotmodel:
                ax.plot(wav_model[i], flux_model[i], 'r-')
                ax.set_xlim([wav_model[i][0],wav_model[i][-1]])
            if plotres and not (plotmodel or plotobs):
                ax.plot(wav_obs[i], flux_obs[i]-flux_model[i], 'k-')
                ax.set_xlim([wav_obs[i][0],wav_obs[i][-1]])
            ax.text(0.03, 0.90, "{}: {}".format(self.data['id'][inds[i]], self.data['phase'][inds[i]]), transform=ax.transAxes, bbox={'facecolor':'w'})
            if plotres and (plotmodel or plotobs):
                ax = fig.add_axes(ax2lim)
                ax.plot(wav_obs[i], flux_obs[i]-flux_model[i], 'k-')
                ax.set_xlim([wav_obs[i][0],wav_obs[i][-1]])
            if (i+1)%(panels[0]*panels[1]) == 0:
                pylab.draw()
                raw_input("Press enter to move to the next plot")
            elif i == len(inds)-1:
                pylab.draw()
        return

    def Plot_data(self, inds=None, verbose=True, panels=None):
        """
        Plots the observed and predicted values along with the light curve.

        inds: Indices of the data to be plotted
        panels (list): The layout of the plot. Should be a list: (ncol, nrows). If none,
            there will be only one spectrum per window.

        >>> self.Plot([PIBYTWO,1.,0.9,4000.,0.08,1.4,0.07,10.,0.])
        """
        if inds is None:
            inds = np.arange(self.ndataset)
        if isinstance(inds, int):
            inds = [inds]

        flux_obs = [self.data['flux'][i] for i in inds]
        flux_obs_err = [self.data['err'][i] for i in inds]
        wave_obs = [self.data['wavelength'][i] for i in inds]

        if len(pylab.get_fignums()) == 0:
            fig = pylab.figure()
        else:
            fig = pylab.gcf()
        if panels is None:
            for i in xrange(len(inds)):
                fig.clf()
                ax1 = fig.add_axes([0.05,0.05,0.92,0.93])
                ax1.plot(wave_obs[i], flux_obs[i], 'k-')
                ax1.set_xlim([wave_obs[i][0],wave_obs[i][-1]])
                ax1.text(0.03, 0.90, "{}: {}".format(self.data['id'][inds[i]], self.data['phase'][inds[i]]), transform=ax1.transAxes, bbox={'facecolor':'w'})
                pylab.draw()
                raw_input("Press enter to move to the next plot")
        else:
            llim = 0.05
            blim = 0.05
            wlim = 0.92
            hlim = 0.93
            colspace = 0.03
            rowspace = 0.03
            split_ratio = 0.3
            w = (wlim - (panels[0]-1)*colspace)/panels[0]
            h = (hlim - (panels[1]-1)*rowspace)/panels[1]
            for i in xrange(len(inds)):
                if i%(panels[0]*panels[1]) == 0:
                    fig.clf()
                col = (i%(panels[0]*panels[1]))/panels[1]
                row = (i%(panels[0]*panels[1]))%panels[1]
                ax1lim = [llim+col*(colspace+w),(blim+(panels[1]-1-row)*(rowspace+h)),w,h]
                ax1 = fig.add_axes(ax1lim)
                ax1.plot(wave_obs[i], flux_obs[i], 'k-')
                ax1.set_xlim([wave_obs[i][0],wave_obs[i][-1]])
                ax1.text(0.03, 0.90, "{}: {}".format(self.data['id'][inds[i]], self.data['phase'][inds[i]]), transform=ax1.transAxes, bbox={'facecolor':'w'})
                if (i+1)%(panels[0]*panels[1]) == 0:
                    pylab.draw()
                    raw_input("Press enter to move to the next plot")
                elif i == len(inds)-1:
                    pylab.draw()
        return

    def Plot_model(self, par, inds=None, verbose=True, panels=None):
        """
        Plots the observed and predicted values along with the light curve.

        par: Parameter list.
            [0]: Orbital inclination in radians.
            [1]: Corotation factor.
            [2]: Roche-lobe filling.
            [3]: Companion temperature.
            [4]: Gravity darkening coefficient.
            [5]: K (projected velocity semi-amplitude) in m/s.
            [6]: Front side temperature.
            [7]: Systematic velocity offset in m/s.
        inds: Indices of the data to be plotted
        panels (list): The layout of the plot. Should be a list: (ncol, nrows). If none,
            there will be only one spectrum per window.

        >>> self.Plot([PIBYTWO,1.,0.9,4000.,0.08,1.4,0.07,10.,0.])
        """
        if inds is None:
            inds = np.arange(self.ndataset)
        if isinstance(inds, int):
            inds = [inds]

        ## Calculating the spectra with the polynomial continuum fitting
        flux_model = self.Get_flux(par, orbph=self.data['phase'][inds], velocities=self.data['v_offset'][inds], verbose=False)
        wave_model = self.atmo_grid.wav

        if len(pylab.get_fignums()) == 0:
            fig = pylab.figure()
        else:
            fig = pylab.gcf()
        if panels is None:
            for i in xrange(len(inds)):
                fig.clf()
                ax1 = fig.add_axes([0.05,0.05,0.92,0.93])
                ax1.plot(wave_model, flux_model[i], 'k-')
                ax1.set_xlim([wave_model[0],wave_model[-1]])
                ax1.set_xlim([wave_model[0],wave_model[-1]])
                ax1.text(0.03, 0.90, "{}: {}".format(self.data['id'][inds[i]], self.data['phase'][inds[i]]), transform=ax1.transAxes, bbox={'facecolor':'w'})
                pylab.draw()
                raw_input("Press enter to move to the next plot")
        else:
            llim = 0.05
            blim = 0.05
            wlim = 0.92
            hlim = 0.93
            colspace = 0.03
            rowspace = 0.03
            split_ratio = 0.3
            w = (wlim - (panels[0]-1)*colspace)/panels[0]
            h = (hlim - (panels[1]-1)*rowspace)/panels[1]
            for i in xrange(len(inds)):
                if i%(panels[0]*panels[1]) == 0:
                    fig.clf()
                col = (i%(panels[0]*panels[1]))/panels[1]
                row = (i%(panels[0]*panels[1]))%panels[1]
                ax1lim = [llim+col*(colspace+w),(blim+(panels[1]-1-row)*(rowspace+h)),w,h]
                ax1 = fig.add_axes(ax1lim)
                ax1.plot(wave_model, flux_model[i], 'k-')
                ax1.set_xlim([wave_model[0],wave_model[-1]])
                ax1.text(0.03, 0.90, "{}: {}".format(self.data['id'][inds[i]], self.data['phase'][inds[i]]), transform=ax1.transAxes, bbox={'facecolor':'w'})
                if (i+1)%(panels[0]*panels[1]) == 0:
                    pylab.draw()
                    raw_input("Press enter to move to the next plot")
                elif i == len(inds)-1:
                    pylab.draw()
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
            self.star.Make_surface(q=q, omega=corot, filling=fill, temp=temp_back, tempgrav=gdark, tirr=tirr, porb=self.porb, k1=K, incl=incl)
        separation = self.star.separation
        roche = self.star.Roche()
        Mwd = self.star.mass1
        Mns = self.star.mass2
        # below we transform sigma from W m^-2 K^-4 to erg s^-1 cm^-2 K^-4
        # below we transform the separation from m to cm
        Lirr = tirr**4 * (cts.sigma*1e3) * (separation*100)**2 * 4*cts.PI
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
            print( "Irration luminosity: %5.4e Lsun" %Lirr )
            print( "Backside temperature: %7.2f K" %temp_back )
            print( "Frontside temperature: %7.2f (tabul.), %7.2f (approx.) K" %(np.exp(self.star.logteff.max()),temp_front) )
            print( "" )
            print( "Inclination: %5.3f rad (%6.2f deg)" %(incl,incl*cts.RADTODEG) )
            print( "System velocity: %7.3f km/s" %(vel_sys/1000) )
            print( "K: %7.3f km/s" %(K/1000) )
            print( "" )
            print( "Mass ratio: %6.3f" %q )
            print( "Mass NS: %5.3f Msun" %Mns )
            print( "Mass Comp: %5.3f Msun" %Mwd )
        return

    def __Read_atmo(self, atmo_fln):
        """__Read_atmo(atmo_fln)
        Reads the atmosphere model data.
        
        atmo_fln: A file containing the grid model information for the whole
            data set. The file contains one line having the following format:
                descriptor name (str): description of the grid
                grid file (str): the grid files to be read; can be a wildcard
                wavelength cut low (float): the lower limit to trim the grid
                wavelength cut high (float): the upper limit to trim the grid
                oversample (int): Oversampling factor. A linear interpolation
                    will be performed in order to oversample the grid in the
                    wavelength dimension by a factor 'oversample'. If 0, no
                    oversampling will be performed.
                smooth (float): A Gaussian smoothing with a sigma equals to
                    'smooth' in the wavelength dimension will be performed.
                    If 0, no smoothing will be performed.
        
        >>> self.__Read_atmo(atmo_fln)
        """
        f = open(atmo_fln,'r')
        lines = f.readlines()
        tmp = lines[0].split()
        flns = glob.glob(tmp[1])
        flns.sort()
        wavelow = float(tmp[2])
        wavehigh = float(tmp[3])
        if len(tmp) > 4:
            oversample = int(tmp[4])
            if oversample == 0:
                oversample = None
        else:
            oversample = None
        sigma = float(tmp[5]) if len(tmp) > 5 else None
        tophat = int(tmp[6]) if len(tmp) > 6 else None
        self.atmo_grid = Atmosphere.Atmo_BTSettl7_spectro(flns, oversample=oversample, sigma=sigma, tophat=tophat, wave_cut=[wavelow, wavehigh])
        return

    def __Read_data(self, data_fln, phase_offset=-0.25, wave_cut=None):
        """__Read_data(self, data_fln, phase_offset=-0.25, wave_cut=None)
        Reads the photometric data.
        
        data_fln (str): A file containing the information for each data set.
            The format of the file is as follows:
                descriptor name
                data file
                column name wavelength
                column name flux
                column name error flux
                orbital phase
                barycenter offset (i.e. measured velocity for optical 
                    companion in km/s)
                barycenter offset error (i.e. measured velocity error 
                    for optical companion in km/s)
                approximate velocity (i.e. measure velocity for pulsar companion
                    with velocity_find in km/s)
        phase_offset (float): Value to be added to the orbital phase in order
            to have phase 0.0 and 0.5 being conjunction times, with 0.0 the eclipse.
        wave_cut (array): Lower and upper wavelength limit 
        
        >>> self.__Read_data(data_fln)
        """
        f = open(data_fln,'r')
        lines = f.readlines()
        self.data = {'wavelength':[], 'flux':[], 'phase':[], 'err':[], 'v_offset':[], 'v_offset_err':[], 'v_approx':[], 'fln':[], 'id':[]}
        for line in lines:
            if not line.startswith('#'):
                tmp = line.split()
                sys.stdout.write( "Reading data file {}\r".format(tmp[0]) ); sys.stdout.flush()
                self.data['id'].append(tmp[0])
                self.data['fln'].append(tmp[1])
                d = np.loadtxt(tmp[1], usecols=[int(tmp[2]),int(tmp[3]),int(tmp[4])], unpack=True)
                wavelow, wavehigh = float(tmp[9]), float(tmp[10])
                if wavelow != wavehigh:
                    inds = (d[0] >= wavelow)*(d[0] <= wavehigh)
                    d = d[:,inds]
                self.data['wavelength'].append(d[0])
                self.data['flux'].append(d[1])
                self.data['err'].append(d[2])
                self.data['phase'].append((float(tmp[5])+phase_offset)%1)
                ## We have to convert the velocities from km/s to m/s
                self.data['v_offset'].append(float(tmp[6]) * 1000)
                self.data['v_offset_err'].append(float(tmp[7]) * 1000)
                self.data['v_approx'].append(float(tmp[8]) * 1000)
        sys.stdout.write("\n"); sys.stdout.flush()
        self.data['id'] = np.asarray(self.data['id'])
        self.data['phase'] = np.asarray(self.data['phase'])
        self.data['v_offset'] = np.asarray(self.data['v_offset'])
        self.data['v_offset_err'] = np.asarray(self.data['v_offset_err'])
        self.data['v_approx'] = np.asarray(self.data['v_approx'])
        return

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
        for i in np.arange(self.ndataset):
            err = np.abs(self.data['err'][i]/self.data['flux'][i]*flux[i])
            np.savetxt(fln+".%02i" %i, np.c_[self.data['wavelength'][i],flux[i],err])
        return

######################## class Spectroscopy ########################



######################## Utility functions for spectroscopy ########################


def Process_flux(flux_obs, flux_obs_err, flux_model, wave_obs, wave_model, z=0., **kwargs):
    """
    Process the model flux and return it along with its chi-square.
    If chi2only = True, returns only the chi-square.

    Rebin the model spectra at the observed wavelengths.
    Convolve the model spectra to match the observed spectra.
    Normalize the model spectra to match the observed spectra.

    z: Doppler shifts to apply, in units of v/c.
    """
    ## Rebin
    if True:
        opts = {}
        if 'interpolate' in kwargs:
            opts['interpolate'] = kwargs['interpolate']
        if 'z' == 0.:
            flux_model = Rebin(flux_model, wave_model, wave_obs, **opts)
        else:
            flux_model = Rebin(flux_model, wave_model*(1+z), wave_obs, **opts)
    ## Convolve
    if False:
        opts = {}
        if 'sigma' in kwargs:
            opts['sigma'] = kwargs['sigma']
        if 'top' in kwargs:
            opts['top'] = kwargs['top']
        flux_model = Convolve_gaussian_tophat(flux_model, **opts)
    ## Normalize
    if True:
        opts = {}
        if 'coeff' in kwargs:
            opts['coeff'] = kwargs['coeff']
        if 'chi2only' in kwargs:
            opts['chi2only'] = kwargs['chi2only']
        result = Normalize_spectrum(flux_model, flux_obs, flux_err=flux_obs_err, **opts)
        return result
    return flux_model

def Process_flux1(flux_obs, flux_obs_err, flux_model, wave_obs, wave_model, v_approx, **kwargs):
    """
    Process the model flux and returns the best velocity and chi-square.

    Tries to find the best velocity using "fmin".

    Rebin the model spectra at the observed wavelengths.
    Convolve the model spectra to match the observed spectra.
    Normalize the model spectra to match the observed spectra.
    """
    kwargs['chi2only'] = True
    def return_chi2(z):
        chi2 = Process_flux(flux_obs, flux_obs_err, new_flux_model, wave_obs, wave_model, z=z **kwargs)
        return np.sum(chi2)
    sol = scipy.optimize.fmin(return_chi2, v_approx, full_output=1)
    return sol[:2]

def Doppler_shift(flux, z, z0=1):
    """Doppler_shift(flux, z, z0=1)
    Shift the spectrum according to the velocity z.
    Assumes that the sampling of the flux is on
    a log(wavelength) scale (linear in velocity).

    flux (array): Spectrum to be resampled.
    z (float): Doppler shift to apply, in units of z0.
        z < 0 means the source is blueshifted.
        z > 0 means the source is redshifted.
    z0 (float): The sampling size of the wavelength.

    Note: This function works on a multi-dimensional flux
        array the last dimension is the wavelength axis.
    """
    ## Convert the shift in units of bin size
    z = -z/z0
    if z == 0:
        newflux = flux
    else:
        ws = z%1
        bin = int(abs(np.floor(z)))
        newflux = np.empty_like(flux)
        if z >= 0:
            if ws == 0:
                newflux[...,:flux.shape[-1]-bin] = flux[...,bin:]
                newflux[...,flux.shape[-1]-bin:] = flux[...,-1]
            else:
                newflux[...,:flux.shape[-1]-bin-1] = flux[...,bin:-1]*(1-ws) + flux[...,bin+1:]*ws
                newflux[...,flux.shape[-1]-bin-1:] = flux[...,-1]
        else:
            bin = abs(bin)
            ws = abs(ws)
            if ws == 0:
                newflux[...,bin:] = flux[...,:flux.shape[-1]-bin]
                newflux[...,:bin] = flux[...,0]
            else:
                newflux[...,bin:] = flux[...,:flux.shape[-1]-bin]*(1-ws) + flux[...,1:flux.shape[-1]-bin+1]*ws
                newflux[...,:bin] = flux[...,0]
    return newflux

def Normalize_spectrum(flux_model, flux, flux_err=None, a_n=None, coeff=3, chi2only=False):
    """
    Normalize a model spectrum to fit an observed one
    using a polynomial fit. Returns the normalized model
    spectrum and its chi-square.

    flux_model (array): Model spectrum to be normalized.
        shape -> (npoints)
    flux (array): Observed spectrum to normalize to.
        shape -> (npoints)
    flux_err (array): Errors on the observed spectrum.
        shape -> None or float or (npoints)
    a_n (array): Polynomial coefficients for the normalization.
        If provided, no fit will have to be done.
        shape -> (ncoeff)
    coeff (int): Polynomial order of the fit to perform.
    chi2only (bool): If True, will return the chi-square only.
    """
    if a_n is None:
        x = np.arange(np.size(flux_model)) - np.size(flux_model)/2
        a_n, chi2 = Utils.Series.GPolynomial_fit(flux, x=x, err=flux_err, coeff=coeff, Xfnct=flux_model, Xfnct_offset=False)
    if chi2only:
        return chi2
    else:
        poly = np.poly1d(a_n)
        norm_flux_model = poly(x) * flux_model
    return norm_flux_model, chi2

def Rebin(flux, x, xnew, interpolate=True):
    """Rebin(flux, x, xnew, interpolate=True)
    Rebin a spectrum from a given sampling (i.e. log(lambda))
    to another one (i.e. lambda).
    
    The function can deal with a single spectrum or a set of
    multiple spectra. In the latter case, the wavelength axis
    must be along the last dimension and the interpolation the
    same of all.

    flux (array): Spectrum to be resampled.
        (nsample) or (nspectra,nsample)
    x (array): Current sampling of the spectrum.
        (nsample)
    xnew (array): New sampling of the spectrum.
        (nsample)
    interpolate (bool): If true, will use the Utils.Series.Interp_linear
        function, which is faster. If false, will use the
        Utils.Series.Interp_integrate function, which is slower but more
        accurate.
    """
    if interpolate:
        w,inds = Utils.Series.Getaxispos_vector(x, xnew)
        newflux = Utils.Series.Interp_linear(flux, w, inds)
    else:
        if np.ndim(flux) == 2:
            newflux = np.empty((np.shape[0],xnew.shape[0]))
            for i in xrange(len(flux)):
                newflux[i] = Utils.Series.Interp_integrate(flux[i], x[i], xnew[i])
        else:
            newflux = Utils.Series.Interp_integrate(flux, x, xnew)
    return newflux

