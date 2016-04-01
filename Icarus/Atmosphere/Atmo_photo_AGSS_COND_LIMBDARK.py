# Licensed under a 3-clause BSD style license - see LICENSE

__all__ = ["Atmo_BTSettl7_spectro", "Read_BTSettl7"]

import sys

from astropy.io import fits

from ..Utils.import_modules import *
from .. import Utils
from .Atmo import Atmo_grid

logger = logging.getLogger(__name__)


######################## class Atmo_photo_AGSS_COND_LIMBDARK ########################
class Atmo_photo_AGSS_COND_LIMBDARK(Atmo_grid):
    """
    This class handles the photometric atmosphere grid from AGSS_COND_LIMBDARK.
    """
    def __init__(self, fln, temp_cut=None, logg_cut=None, zp=0., verbose=False):
        """
        """
        self.zp = zp
        self.fln = fln
        self.Flux_init(flns, temp_cut=temp_cut, logg_cut=logg_cut, verbose=verbose)

    def Flux_init(self, flns, temp_cut=None, logg_cut=None, verbose=False):
        """
        flns (list): Input filenames.
        temp_cut (list): Allows to define a lower and an upper limit to the temperature.
        logg_cut (list): Allows to define a lower and an upper limit to the logg.
        verbose (bool): Verbosity.
        
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

######################## class Atmo_photo_AGSS_COND_LIMBDARK ########################



######################## Utilities ########################
def Read(fln):
    """Read(fln)
    Reads a band file and return the flux.
    
    fln: filename
    
    >>> grid, wav, z = Read_original(fln)
    """
    ## Load the hdu
    hdu = fits.open(fln)

    ## The flux grid. Original values are [cgs], so we convert to log10([cgs])
    grid = np.log10(hdu[0].data)

    ## The mu grid
    mu = hdu[1].data

    ## Extra parameters
    temp = hdu[0].header['PHXTEFF']
    logg = np.log(hdu[0].header['PHXLOGG'])

    return logtemp, logg, mu, grid

def Read_multiple(flns, temp_cut=None, logg_cut=None, verbose=False):
    """
    flns (list): Input filenames.
    temp_cut (list): Allows to define a lower and an upper limit to the temperature.
    logg_cut (list): Allows to define a lower and an upper limit to the logg.
    verbose (bool): Verbosity.
    
    >>> Read_multiple(flns)
    """
    temp = []
    logg = []
    mu = []
    
    ## Reading the parameter information about the spectra
    lst = []
    for i in np.arange(len(flns)):
        ## Read the data
        temp_i, logg_i, mu_i, grid_i = Read(fln)
        if temp_cut is None or (temp_i >= temp_cut[0] and temp_i <= temp_cut[1]):
            if logg_cut is None or (logg_i >= logg_cut[0] and logg_i <= logg_cut[1]):
                lst.append( [temp_i, logg_i, mu_i, grid_i] )
                if temp_i not in self.logtemp:
                    temp.append(temp_i)
                if logg_i not in self.logg:
                    logg.append(logg_i)

    ## Ordering the values
    temp.sort()
    logg.sort()
    mu = lst[-1][2]

    ## Putting the flux grid in the right order
    grid = np.empty((len(temp),len(logg),len(mu)), dtype=float)
    for i,temp_i in enumerate(temp):
        for j,logg_i in enumerate(logg):
            found = False
            for l in lst:
                if l[0] == temp_i and l[1] == logg_i:
                    grid[i,j] = l[3]
                    found = True
            if not found:
                print("Missing -> temp: {:5.0f}, logg: {:3.1f}".format(temp_i,logg_i))
                raise Exception( "There is a mismatch in the number of log(g) and temp grid points!" )

    ## Make everything numpy arrays
    logtemp = np.log(temp)
    logg = np.array(logg)
    mu = np.array(mu)
    grid = np.array(grid)

    ## Save to fits table
    return





