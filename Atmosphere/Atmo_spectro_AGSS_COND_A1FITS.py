# Licensed under a 3-clause BSD style license - see LICENSE

__all__ = ["Atmo_AGSS_spectro", "Read_AGSS"]

try:
    import astropy.io.fits as pyfits
except:
    print("Cannot find an installation of astropy to load the fits IO module.")
    print("Trying to load pyfits as a standalone module instead.")
    try:
        import pyfits
    except:
        print("Cannot find an installation of pyfits neither.")
        print("This module (Atmo_AGSS_spectro) will not work properly.")

from ..Utils.import_modules import *
from .. import Utils
from .Atmo import Atmo_grid

logger = logging.getLogger(__name__)


######################## class Atmo_AGSS_spectro ########################
class Atmo_AGSS_spectro(Atmo_grid):
    """
    This class handles the atmosphere grid containing a spectral
    dimension.
    """
    def __init__(self, flns, oversample=None, sigma=None, tophat=None, thin=None, convert=None, zp=0., wave_cut=[3000,11000], temp_cut=None, logg_cut=None, linlog=False, verbose=False, savememory=True):
        """__init__
        """
        # zp is for compatibility of spectroscopic and photometric atmosphere grids and scales the
        # zero point for the magnitude conversion.
        self.zp = zp
        self.flns = flns
        self.Flux_init(flns, oversample=oversample, sigma=sigma, tophat=tophat, thin=thin, convert=convert, wave_cut=wave_cut, temp_cut=temp_cut, logg_cut=logg_cut, linlog=linlog, verbose=verbose)

    def Flux_init(self, flns, oversample=None, sigma=None, tophat=None, thin=None, wave_cut=None, temp_cut=None, logg_cut=None, convert=None, linlog=False, verbose=False):
        """
        Reads atmosphere model files and construct a grid.
        Calculates:
            teff: effective temperatures. teff.shape = (nteff)
            logg: log of surface gravity. logg.shape = (nlogg)
            mu: cos(angle) of emission direction. mu.shape = (nmu)
            grid: the grid of specific intensities. grid.shape = (nteff,nlogg,nmu)
            leff: ???
            h: ???

        fln: filenames
        oversample (int): Oversampling factor (integer). If provided, a cubic spline
            interpolation will be performed in order to oversample the grid in the
            wavelength dimension by a factor 'oversample'.
        sigma (None): If provided, the grid will be smoothed with a Gaussian where the
            sigma is provided in wavelength units. This should account for the seeing.
        tophat (None): If provided, the grid will be smoothed with a tophat function
            where the width is provided in wavelength units. This should account for
            the slit width.
        thin (int): Thinning factor (integer). If provided, the grid will be thinned
            by keeping every other 'thin' values in the wavelength dimension.
        wave_cut (list): Allows to define a lower-upper cut in wavelength [wave_low, wave_up].
        temp_cut (list): Allows to define a lower and an upper limit to the temperature.
        logg_cut (list): Allows to define a lower and an upper limit to the logg.
        convert (str): If not None, will append 'convert' at the end of the filename
            and save the results therein.
        linlog (bool): If true, will rebin the data to be linear in the log space.
        verbose (bool): verbosity.
        
        >>> self.Flux_init(flns)
        """
        ## Reading the parameter information about the spectra
        lst = []
        for i in numpy.arange(len(flns)):
            ## Get the logg and temp value from the filename
            hdr = pyfits.getheader(flns[i], ext=0)
            temp = hdr['PHXTEFF']
            logg = hdr['PHXLOGG']
            if temp_cut is None or (temp >= temp_cut[0] and temp <= temp_cut[1]):
                if logg_cut is None or (logg >= logg_cut[0] and logg <= logg_cut[1]):
                    lst.append( [i, logg, temp] )

        ## Reading the mu values
        self.mu = numpy.array(pyfits.getdata(flns[0], ext=1), dtype=float)
        n_mu = self.mu.size

        ## Sorting the grid by temperature and then logg
        Utils.Misc.Sort_list(lst, [2,1])
        lst = numpy.array(lst)

        ## Extracting the temperature values
        self.logtemp = numpy.log(numpy.unique(lst[:,2]))
        self.logtemp.sort()
        n_teff = self.logtemp.size

        ## Extracting the logg values
        self.logg = numpy.unique(lst[:,1])
        self.logg.sort()
        n_logg = self.logg.size

        ## If there is a mismatch and the grid is not rectangular, then the function aborts
        if n_teff*n_logg != lst.shape[0]:
            print( "Number of temperature points: {}".format(n_teff) )
            print( "Number of logg points: {}".format(n_logg) )
            print( "Number of grid points: {}".format(lst.shape[0]) )
            for teff in self.logtemp:
                for logg in self.logg:
                    missing = True
                    for l in lst:
                        if numpy.log(l[2]) == teff and l[1] == logg:
                            missing = False
                    if missing:
                        print("Missing -> logg: {:3.1f}, temp: {:5.0f}".format(logg,numpy.exp(teff)))
            raise Exception( "There is a mismatch in the number of log(g) and teff grid points!" )
            return

        ## Extracting the data
        grid = []
        wav = []
        if verbose: print( "Starting to read atmosphere grid files" )
        for i,l in enumerate(lst[:,0]):
            if verbose: sys.stdout.write( "Reading {} ({}/{})\r".format(flns[int(l)], i+1, lst.shape[0]) ); sys.stdout.flush()
            tmp = Read_AGSS(flns[int(l)], oversample=oversample, sigma=sigma, tophat=tophat, thin=thin, wave_cut=wave_cut, convert=convert, linlog=linlog)
            grid.append(tmp[0])
            wav.append(tmp[1])
            self.z0 = tmp[2]
            logger.log(8, "Number of wavelength points: {}, range: [{}, {}]".format(tmp[1].size, tmp[1][0], tmp[1][-1]) )
        if verbose: print( "\nFinished reading atmosphere grid files" )
        try:
            wav = numpy.array(wav)
            if wav.std(0).max() > 1.e-6:
                raise Exception( "The wavelength grid is not uniform!" )
                return
            else:
                wav = wav[0]
        except:
            raise Exception( "The wavelength grid has an inconsistent number of elements!" )
            return
        if verbose: print( "Transforming grid data to array" )
        grid = numpy.asarray(grid)
        if verbose: print( "Addressing the grid data shape" )
        grid.shape = n_teff, n_logg, n_mu, wav.size
        self.wav = wav
        if verbose: print( "Making the grid a class attribute" )
        self.grid = grid

        ## Calculating the grid log-to-linear weights
        if linlog:
            self.wav_linear = Utils.Series.Resample_loglin(self.wav)
            self.wav_delta = self.wav_linear[1] - self.wav_linear[0]
            self.wav_frac, self.wav_inds = Utils.Series.Getaxispos_vector(self.wav, self.wav_linear)
        return

    def Get_flux_doppler(self, val_logtemp, val_logg, val_mu, val_area, val_vel):
        """
        Returns the flux interpolated from the atmosphere grid.
        val_logtemp: log of effective temperature
        val_logg: log of surface gravity
        val_mu: cos(angle) of angle of emission
        val_area: area of the surface element
        val_vel: velocity of the grid point in units of speed of light
        
        >>> flux = self.Get_flux_doppler(val_logtemp, val_logg, val_mu, val_area, val_vel)
        """
        logger.debug("start")
        grid = self.grid
        logtemp = self.logtemp
        logg = self.logg
        mu = self.mu
        logger.log(9, "Getting temp indices")
        logger.log(5, "logtemp.min() {}, logtemp.max() {}".format(numpy.min(logtemp),numpy.max(logtemp)) )
        logger.log(5, "val_logtemp.min() {}, val_logtemp.max() {}".format(numpy.min(val_logtemp),numpy.max(val_logtemp)) )
        wtemp, jtemp = self.Getaxispos(logtemp,val_logtemp)
        logger.log(9, "Getting logg indices")
        wlogg, jlogg = self.Getaxispos(logg,val_logg)
        logger.log(9, "Getting mu indices")
        wmu, jmu = self.Getaxispos(mu,val_mu)
        #wmu = numpy.ones_like(wmu)
        #jmu = numpy.zeros_like(jmu, dtype=int)+76

        if self.z0:
            val_vel = val_vel/(-self.z0)
            wwav = numpy.remainder(val_vel, 1)
            jwav = numpy.floor(val_vel).astype(int)
            #Interp_doppler(grid, wteff, wlogg, wmu, wwav, jteff, jlogg, jmu, jwav, area, val_mu)
            #Interp_doppler_savememory(grid, wteff, wlogg, wmu, wwav, jteff, jlogg, jmu, jwav, mu_grid, area, val_mu)
            flux = Utils.Grid.Interp_doppler(grid, wtemp, wlogg, wmu, wwav, jtemp, jlogg, jmu, jwav, val_area, val_mu)
        else:
            print( 'Hey! Wake up! The grid is not linear in lambda and has been transformed to linear in log(lambda)!' )

        logger.debug("end")
        return flux

    def Resample_loglin(self, flux):
        """
        Resample the 'flux' from logarithmic to linear wavelength sampling.

        flux (array): Can be a 1d array, or a 2d array (nflux, nwav).
        """
        return Utils.Series.Interp_linear(flux, self.wav_frac, self.wav_inds)
######################## class Atmo_AGSS_spectro ########################



######################## Utilities for Atmo_AGSS ########################
def Read_AGSS(fln, oversample=None, sigma=None, tophat=None, thin=None, wave_cut=None, convert=None, linlog=False):
    """Read_AGSS(fln, oversample=None, sigma=None, tophat=None, thin=None, wave_cut=None, convert=None, linlog=False)
    Reads a band file and return the grid and wavelength.
    
    fln: filename
    oversample (None): Oversampling factor (integer). If provided, a cubic spline
        interpolation will be performed in order to oversample the grid in the
        wavelength dimension by a factor 'oversample'.
    sigma (None): If provided, the grid will be smoothed with a Gaussian where the
        sigma is provided in wavelength units. This should account for the seeing.
    tophat (None): If provided, the grid will be smoothed with a tophat function
        where the width is provided in wavelength units. This should account for
        the slit width.
    thin (None): Thinning factor (integer). If provided, the grid will be thinned
        by keeping every other 'thin' values in the wavelength dimension.
    wave_cut (None): Allows to define a lower-upper cut in wavelength [wave_low, wave_up].
    convert (None): If not None, will append 'convert' at the end of the filename
        and save the results therein.
    linlog (False): If true, will rebin the data to be linear in the log space.
    
    >>> grid, wav, z = Read_BTSettl7(fln, thin=20)
    """
    ## Opening the file table
    hdu = pyfits.open(fln)

    ## Extracting the wavelength from the header
    hdr = hdu[0].header
    wav = hdr['CRVAL1'] + numpy.arange(hdr['NAXIS1'], dtype=float) * hdr['CDELT1']

    ## Extracting the data. grid.shape = n_mu, n_wavelength
    grid = hdu[0].data

    ## Trim the unwanted wavelength range
    if wave_cut is not None:
        inds = (wav >= wave_cut[0]) * (wav <= wave_cut[1])
        grid = grid[:,inds]
        wav = wav[inds]

    ## Oversample the spectrum if requested
    if oversample is not None and oversample != 1:
        #grid = scipy.ndimage.zoom(grid, oversample, order=1, mode='reflect')
        #wav = numpy.linspace(wav[0], wav[-1], wav.size*oversample)
        interp = scipy.interpolate.UnivariateSpline(wav, grid, k=1, s=0)
        wav = numpy.linspace(wav[0], wav[-1], wav.size*oversample+1)
        grid = interp(wav)

    ## Smooth the spectrum if requested
    logger.log(6,"Original: sigma {}, tophat {}".format(sigma,tophat))
    if sigma is not None or tophat is not None:
        bin = wav[1]-wav[0]
        ## We have to convert values to bin units
        if sigma is None:
            sigma = 0.
        else:
            sigma = sigma/bin
        if tophat is None:
            tophat = 1
        else:
            tophat = int(tophat/bin + 0.5)
            tophat = 1 if tophat < 1 else tophat
        logger.log(6,"Bin converted: bin {}, sigma {}, tophat {}".format(bin,sigma,tophat))
        grid = Utils.Series.Convolve_gaussian_tophat(grid, sigma=sigma, top=tophat)

    ## Thin the spectrum if requested
    if thin is not None:
        grid = grid[::thin]
        wav = wav[::thin]

    ## Convert to logarithmic (velocity) scale so that Doppler boosting is linear
    if linlog:
        new_wav, z = Utils.Series.Resample_linlog(wav)
        ws, inds = Utils.Series.Getaxispos_vector(wav, new_wav)
        wav = new_wav
        grid = grid.take(inds, axis=-1)*(1-ws) + grid.take(inds+1, axis=-1)*ws
    else:
        z = None
    if convert is not None:
        print( "Saving the data into "+fln+convert )
        numpy.savetxt(fln+convert,numpy.vstack((wav,numpy.log10(grid))).T)
    return grid, wav, z

