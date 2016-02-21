# Licensed under a 3-clause BSD style license - see LICENSE

__all__ = ["Atmo_BTSettl7_spectro", "Read_BTSettl7"]

import sys

from ..Utils.import_modules import *
from .. import Utils
from .Atmo import Atmo_grid

logger = logging.getLogger(__name__)


######################## class Atmo_BTSettl7_spectro ########################
class Atmo_spectro_BTSettl7(Atmo_grid):
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
        self.Coeff_limb_darkening(self.wav/1e4, verbose=verbose)
        self.Make_limb_grid(verbose=verbose, savememory=savememory)

    def Coeff_limb_darkening(self, wav, verbose=False):
        """
        Calculates the limb darkening coefficients.
        wav: wavelength in micrometer.
        verbose (=False): verbosity.
        
        Note: Only valid for 0.42257 < wav < 1.100 micrometer.
        From Neckel 2005.
        """
        if verbose: print( "Calculating limb darkening coefficients" )
        def L_422_1100(wav):
            a_00 = 0.75267
            a_01 = -0.265577
            a_10 = 0.93874
            a_11 = 0.265577
            a_15 = -0.004095
            a_20 = -1.89287
            a_25 = 0.012582
            a_30 = 2.42234
            a_35 = -0.017117
            a_40 = -1.71150
            a_45 = 0.011977
            a_50 = 0.49062
            a_55 = -0.003347
            wav5 = wav**5
            a_0 = a_00 + a_01/wav
            a_1 = a_10 + a_11/wav + a_15/wav5
            a_2 = a_20 + a_25/wav5
            a_3 = a_30 + a_35/wav5
            a_4 = a_40 + a_45/wav5
            a_5 = a_50 + a_55/wav5
            return a_0, a_1, a_2, a_3, a_4, a_5
        def L_385_422(wav):
            a_00 = 0.09900
            a_01 = 0.010643
            a_10 = 1.96884
            a_11 = -0.010643
            a_15 = -0.009166
            a_20 = -2.80511
            a_25 = 0.024873
            a_30 = 3.32780
            a_35 = -0.029317
            a_40 = -2.17878
            a_45 = 0.018273
            a_50 = 0.58825
            a_55 = -0.004663
            wav5 = wav**5
            a_0 = a_00 + a_01/wav
            a_1 = a_10 + a_11/wav + a_15/wav5
            a_2 = a_20 + a_25/wav5
            a_3 = a_30 + a_35/wav5
            a_4 = a_40 + a_45/wav5
            a_5 = a_50 + a_55/wav5
            return a_0, a_1, a_2, a_3, a_4, a_5
        def L_300_372(wav):
            a_00 = 0.35601
            a_01 = -0.085217
            a_10 = 1.11529
            a_11 = 0.085217
            a_15 = -0.001871
            a_20 = -0.67237
            a_25 = 0.003589
            a_30 = 0.18696
            a_35 = -0.002415
            a_40 = 0.00038
            a_45 = 0.000897
            a_50 = 0.01373
            a_55 = -0.000200
            wav5 = wav**5
            a_0 = a_00 + a_01/wav
            a_1 = a_10 + a_11/wav + a_15/wav5
            a_2 = a_20 + a_25/wav5
            a_3 = a_30 + a_35/wav5
            a_4 = a_40 + a_45/wav5
            a_5 = a_50 + a_55/wav5
            return a_0, a_1, a_2, a_3, a_4, a_5
        self.limb = np.empty((6,wav.shape[0]))
        inds = wav<0.37298
        self.limb[:,inds] = L_300_372(wav[inds])
        inds = (wav<0.42257)*(wav>0.37298)
        self.limb[:,inds] = L_385_422(wav[inds])
        inds = wav>0.42257
        self.limb[:,inds] = L_422_1100(wav[inds])
        return

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
        for i in np.arange(len(flns)):
            ## Get the logg and teff value from the filename
            tmp = flns[i].split('lte')[1]
            temp = float(tmp[:3])*100
            logg = -float(tmp[3:7])
            if temp_cut is None or (temp >= temp_cut[0] and temp <= temp_cut[1]):
                if logg_cut is None or (logg >= logg_cut[0] and logg <= logg_cut[1]):
                    lst.append( [i, logg, temp] )

        ## Sorting the grid by temperature and then logg
        Utils.Misc.Sort_list(lst, [2,1])
        lst = np.array(lst)

        ## Extracting the temperature values
        self.logtemp = np.log(np.unique(lst[:,2]))
        self.logtemp.sort()
        n_teff = self.logtemp.size

        ## Extracting the logg values
        self.logg = np.unique(lst[:,1])
        self.logg.sort()
        n_logg = self.logg.shape[0]

        ## If there is a mismatch and the grid is not rectangular, then the function aborts
        if n_teff*n_logg != lst.shape[0]:
            print( "Number of temperature points: {}".format(n_teff) )
            print( "Number of logg points: {}".format(n_logg) )
            print( "Number of grid points: {}".format(lst.shape[0]) )
            for teff in self.logtemp:
                for logg in self.logg:
                    missing = True
                    for l in lst:
                        if np.log(l[2]) == teff and l[1] == logg:
                            missing = False
                    if missing:
                        print("Missing -> logg: {:3.1f}, temp: {:5.0f}".format(logg,np.exp(teff)))
            raise Exception( "There is a mismatch in the number of log(g) and teff grid points!" )
            return

        ## Extracting the data
        grid = []
        wav = []
        if verbose: print( "Starting to read atmosphere grid files" )
        for i,l in enumerate(lst[:,0]):
            if verbose: sys.stdout.write( "Reading {} ({}/{})\r".format(flns[int(l)], i+1, lst.shape[0]) ); sys.stdout.flush()
            tmp = Read_BTSettl7(flns[int(l)], oversample=oversample, sigma=sigma, tophat=tophat, thin=thin, wave_cut=wave_cut, convert=convert, linlog=linlog)
            grid.append(tmp[0])
            wav.append(tmp[1])
            self.z0 = tmp[2]
            logger.log(8, "Number of wavelength points: {}, range: [{}, {}]".format(tmp[1].size, tmp[1][0], tmp[1][-1]) )
        if verbose: print( "\nFinished reading atmosphere grid files" )
        try:
            wav = np.array(wav)
            if wav.std(0).max() > 1.e-6:
                raise Exception( "The wavelength grid is not uniform!" )
                return
            else:
                wav = wav[0]
        except:
            raise Exception( "The wavelength grid has an inconsistent number of elements!" )
            return
        if verbose: print( "Transforming grid data to array" )
        grid = np.asarray(grid)
        if verbose: print( "Addressing the grid data shape" )
        grid.shape = n_teff, n_logg, wav.size
        self.wav = wav
        if verbose: print( "Making the grid a class attribute" )
        self.grid = grid

        ## Calculating the grid log-to-linear weights
        if linlog:
            self.wav_linear = Utils.Series.Resample_loglin(self.wav)
            self.wav_delta = self.wav_linear[1] - self.wav_linear[0]
            self.wav_frac, self.wav_inds = Utils.Series.Getaxispos_vector(self.wav, self.wav_linear)
        return

    def Get_flux_doppler(self, val_logtemp, val_logg, val_mu, val_area, val_vel, **kwargs):
        """
        Returns the flux interpolated from the atmosphere grid.
        val_logtemp: log of effective temperature
        val_logg: log of surface gravity
        val_mu: cos(angle) of angle of emission
        val_area: area of the surface element
        val_vel: velocity of the grid point in units of speed of light
        
        >>> flux = self.Get_flux_doppler(val_logtemp, val_logg, val_mu, val_area, val_vel)
        """
        logger.log(9, "start")
        grid = self.grid
        logtemp = self.logtemp
        logg = self.logg
        mu = self.mu
        logger.log(9, "Getting temp indices")
        logger.log(5, "logtemp.min() {}, logtemp.max() {}".format(np.min(logtemp),np.max(logtemp)) )
        logger.log(5, "val_logtemp.min() {}, val_logtemp.max() {}".format(np.min(val_logtemp),np.max(val_logtemp)) )
        wtemp, jtemp = self.Getaxispos(logtemp,val_logtemp)
        logger.log(9, "Getting logg indices")
        wlogg, jlogg = self.Getaxispos(logg,val_logg)
        logger.log(9, "Getting mu indices")
        wmu, jmu = self.Getaxispos(mu,val_mu)
        if self.savememory:
            grid_mu = self.grid_mu
            if self.z0:
                val_vel = val_vel/(-self.z0)
                wwav = np.remainder(val_vel, 1)
                jwav = np.floor(val_vel).astype(int)
                flux = Utils.Grid.Interp_doppler_savememory(grid, wtemp, wlogg, wmu, wwav, jtemp, jlogg, jmu, jwav, grid_mu, val_area, val_mu)
            else:
                print( 'Hey! Wake up! The grid is linear in lambda and should have been transformed to linear in log(lambda)!' )
        else:
            print( 'This option (savememory=False) is not available at the moment!' )
            #flux = Utils.Grid.Interp_doppler(grid, wtemp, wlogg, wmu, wwav, jtemp, jlogg, jmu, jwav, val_area, val_mu)
        #return flux, wtemp, wlogg, wmu, wwav, jtemp, jlogg, jmu, jwav, grid_mu, val_area, val_mu
        logger.log(9, "end")
        return flux

    def Limb_darkening(self, mu):
        """
        Returns the limb darkening for each wavelength of the grid.
        mu: cos(theta) direction of emission angle.
        
        Note: Only valid for 0.42257 < wav < 1.100 micrometer.
        From Neckel 2005.
        """
        return np.round(self.limb[0] + (self.limb[1] + (self.limb[2] + (self.limb[3] + (self.limb[4] + self.limb[5]*mu )*mu )*mu )*mu )*mu, decimals=10)

    def Make_limb_grid(self, verbose=False, savememory=True):
        """
        Calculates grids for different mu values.
        It is faster to interpolate from a grid than calculating
        the exact flux value every time.
        
        savememory (=False): If true, will keep the mu factors on the
            side and will account for them at the flux calculation time
            in the modified Interp function.
        verbose (=False): verbosity.
        """
        self.savememory = savememory
        if verbose: print( "Calculating the limb darkening grid" )
        self.mu = np.arange(0.,1.05,0.05)
        grid_mu = self.mu.copy()
        grid_mu.shape = grid_mu.size,1
        grid_mu = self.Limb_darkening(grid_mu)
        if self.savememory:
            self.grid_mu = grid_mu
        else:
            g = np.array([self.grid * m for m in mu])
            self.grid = np.ascontiguousarray(g.swapaxes(0,1).swapaxes(1,2))

    def Resample_loglin(self, flux):
        """
        Resample the 'flux' from logarithmic to linear wavelength sampling.

        flux (array): Can be a 1d array, or a 2d array (nflux, nwav).
        """
        return Utils.Series.Interp_linear(flux, self.wav_frac, self.wav_inds)
######################## class Atmo_BTSettl7_spectro ########################



######################## Utilities for Atmo_spectro_BTSettl7 ########################
def Read_BTSettl7(fln, oversample=None, sigma=None, tophat=None, thin=None, wave_cut=None, convert=None, linlog=False):
    """Read_BTSettl7(fln, oversample=None, sigma=None, tophat=None, thin=None, wave_cut=None, convert=None, linlog=False)
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
    ## The wavelength is contained in the first column and the flux in the second.
    wav, grid = np.loadtxt(fln, usecols=(0,1), unpack=True)

    ## Grid values are log10(F_lambda) [cgs]
    #grid = 10**grid * 4/cts.PI**2 # Conversion to flux units (make sure that in the Get_flux routine does not re-correct again!)
    grid = (2 / cts.PI / np.sqrt(3)) * 10**grid

    ## Wavelengths are often not ordered so we re-order them
    inds = wav.argsort()
    wav = wav[inds]
    grid = grid[inds]

    ## Trim the unwanted wavelength range
    if wave_cut is not None:
        inds = (wav >= wave_cut[0]) * (wav <= wave_cut[1])
        grid = grid[inds]
        wav = wav[inds]

    ## Oversample the spectrum if requested
    if oversample is not None and oversample != 1:
        #grid = scipy.ndimage.zoom(grid, oversample, order=1, mode='reflect')
        #wav = np.linspace(wav[0], wav[-1], wav.size*oversample)
        interp = scipy.interpolate.UnivariateSpline(wav, grid, k=1, s=0)
        wav = np.linspace(wav[0], wav[-1], wav.size*oversample+1)
        grid = interp(wav)

    ## Smooth the spectrum if requested
    logger.log(6, "Original: sigma {}, tophat {}".format(sigma,tophat))
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
        logger.log(6, "Bin converted: bin {}, sigma {}, tophat {}".format(bin,sigma,tophat))
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
        np.savetxt(fln+convert,np.vstack((wav,np.log10(grid))).T)
    return grid, wav, z


