# Licensed under a 3-clause BSD style license - see LICENSE

__all__ = ["Atmo_grid_BTSettl7"]

from ..Utils.import_modules import *
from .. import Utils
from .Atmo_grid import Atmo_grid


######################## class Atmo_grid_BTSettl7 ########################
class Atmo_grid_BTSettl7(Atmo_grid):
    """Atmo_grid_BTSettl7(Atmo_grid)
    This class handles the atmosphere grid for photometric data
    from the BT-Settl.7 atmosphere models.
    """
    def __init__(self, fln, lam, dlam, flux0, ext=0., logg_lims=[3.,5.], AB=True):
        """__init__(self, fln, lam, dlam, flux0, ext=0., logg_lims=[3.,5.], AB=True)
        """
        self.grid_lam = lam
        self.dlam = dlam
        self.flux0 = flux0
        self.ext = ext
        self.fln = fln
        self.Flux_init(logg_lims=logg_lims, AB=AB)

    def Flux_init(self, logg_lims=[3.,5.], AB=True):
        """Flux_init(logg_lims=[3.,5.], AB=True)
        Reads a band file and construct a grid.
        
        The data files are: teff, logg, flux.
        
        logg_lims ([3.,5.]): provides the min and max value of logg.
        AB (True): If the grid is in the AB system, only a multiplication by PI needs to be done.
            Otherwise, needs to multiply by c*100/lambda**2.
        
        Calculates:
            grid_teff: effective temperatures. grid_teff.shape = (nteff)
            grid_logg: log of surface gravity. grid_logg.shape = (nlogg)
            grid_mu: cos(angle) of emission direction. grid_mu.shape = (nmu)
            grid: the grid of specific intensities. grid.shape = (nteff,nlogg,nmu)
        
        >>> self.Flux_init(logg_lims=[3.,5.])
        """
        # Load the data
        teff,logg,flux = numpy.loadtxt(self.fln, unpack=True)
        # Extract the unique values of logg
        grid_logg = numpy.unique(logg)
        grid_logg = grid_logg[(grid_logg>=logg_lims[0])*(grid_logg<=logg_lims[1])]
        n_logg = grid_logg.size
        
        # Trim the atmosphere grid to keep only the elements within the logg subset
        inds = numpy.zeros(teff.size, dtype=bool)
        for i in xrange(grid_logg.size):
            inds += logg == grid_logg[i]
        
        teff = teff[inds]
        logg = logg[inds]
        flux = flux[inds]
        
        # Trim the atmosphere grid to keep only the elements within the teff subset
        inds = numpy.zeros(teff.size, dtype=bool)
        grid_teff = []
        for t in numpy.unique(teff):
            if (teff == t).sum() == n_logg:
                grid_teff.append( t )
                inds += teff == t
        
        grid_teff = numpy.array(grid_teff)
        n_teff = grid_teff.size
        teff = teff[inds]
        logg = logg[inds]
        flux = flux[inds]
        
        if teff.size != n_teff*n_logg:
            print( "The number of entries does not constitute a proper grid! (n_total, n_teff, n_logg)", teff.size, n_teff, n_logg )
        
        # Sorting the grid
        inds = numpy.lexsort((logg,teff))
        
        teff.shape = n_teff, n_logg
        logg.shape = n_teff, n_logg
        flux.shape = n_teff, n_logg
        
        # Creating the mu values
        grid_mu = numpy.linspace(0., 1., 16)
        n_mu = 16
        mu = grid_mu.copy()
        mu.shape = n_mu,1
        mu_factor = Utils.Limb_darkening(numpy.r_[self.grid_lam]*1e4, mu)[:,0]
        #mu_factor = numpy.ones_like(mu_factor)
        grid = numpy.resize(flux, (n_mu, n_teff, n_logg))
        grid = grid.swapaxes(0, 1).swapaxes(1, 2) * mu_factor
        if AB:
            grid = numpy.log(grid*4/PI**2)
        else:
            grid = numpy.log(grid*4/PI**2 / (C*1e10) * (self.grid_lam*1e8)**2)
        
        # Making class variables
        self.grid = grid.copy()
        self.grid_teff = numpy.log(grid_teff)
        self.grid_logg = grid_logg.copy()
        self.grid_mu = grid_mu.copy()
        return

######################## class Atmo_grid_BTSettl7 ########################

