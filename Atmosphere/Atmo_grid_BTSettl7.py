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
        n_mu = 16
        grid_mu = numpy.linspace(0., 1., 16)
        
        # Making class variables
        self.grid_teff = numpy.log(grid_teff)
        self.grid_logg = grid_logg.copy()
        self.grid_mu = grid_mu.copy()
        
        # Trying to calculate the limb darkening using Limb_darkening()
        use_claret = False
        if use_claret:
            try:
                print( "Using Claret & Bloemen 2011 for the limb darkening" )
                mu_factor = self.Limb_darkening()
            except:
                print( "Problem calculating the limb darkening from Claret & Bloemen 2011. Using fallback formula." )
                mu = grid_mu.copy()
                mu.shape = n_mu,1
                mu_factor = Utils.Limb_darkening(numpy.r_[self.grid_lam]*1e4, mu)[:,0]
                #mu_factor = numpy.ones_like(mu_factor)
        else:
            mu = grid_mu.copy()
            mu.shape = n_mu,1
            mu_factor = Utils.Limb_darkening(numpy.r_[self.grid_lam]*1e4, mu)[:,0]
        
        # Applying the limb darkening
        grid = numpy.resize(flux, (n_mu, n_teff, n_logg))
        grid = grid.swapaxes(0, 1).swapaxes(1, 2) * mu_factor
        if AB:
            grid = numpy.log(grid*4/PI**2)
        else:
            grid = numpy.log(grid*4/PI**2 / (C*1e10) * (self.grid_lam*1e8)**2)
        
        # Making class variables
        self.grid = grid.copy()
        
        return

    def Limb_darkening(self):
        """Limb_darkening()
        Calculates the limb darkening coefficients using the coefficients from
        Claret & Bloemen 2011.
        
        >>> self.Limb_darkening()
        """
        # Making the filename to extract the limb darkening coefficients
        fln = self.fln.replace('BT-Settl.7', 'claret_2011/PHOENIX/ldc')
        # Reading the file. Format: 0: logg, 1: temp, 2,3,4,5: coefficients
        d = numpy.loadtxt(fln, usecols=(0,1,4,5,6,7), unpack=True)
        
        # Determining the range of logg and temperature, making sure everything is fine.
        logg = numpy.unique(d[0])
        if d[0].size%logg.size != 0:
            #print( "The number of logg values is odd and does not match the total number of entries." )
            raise RuntimeError( "The number of logg values is odd and does not match the total number of entries." )
        teff = numpy.unique(d[1])
        if d[1].size%teff.size != 0:
            #print( "The number of temp values is odd and does not match the total number of entries." )
            raise RuntimeError( "The number of temp values is odd and does not match the total number of entries." )
        if d[0].size != logg.size*teff.size:
            #print( "The total number of entries should equal the number of logg times the number of temp." )
            raise RuntimeError( "The total number of entries should equal the number of logg times the number of temp." )
        if logg.min() > self.grid_logg.min() or logg.max() < self.grid_logg.max():
            print( "Careful! The min/max of the limb darkening are {}/{}, versus those of the atmosphere grid {}/{}".format(logg.min(), logg.max(), self.grid_logg.min(), self.grid_logg.max()) )
            #raise RuntimeError( "Careful! The min/max of the limb darkening are {}/{}, versus those of the atmosphere grid {}/{}".format(logg.min(), logg.max(), self.grid_logg.min(), self.grid_logg.max()) )
        if teff.min() > numpy.exp(self.grid_teff.min()) or teff.max() < numpy.exp(self.grid_teff.max()):
            print( "Careful! The min/max of the limb darkening are {}/{}, versus those of the atmosphere grid {}/{}".format(teff.min(), teff.max(), numpy.exp(self.grid_teff.min()), numpy.exp(self.grid_teff.max())) )
            #raise RuntimeError( "Careful! The min/max of the limb darkening are {}/{}, versus those of the atmosphere grid {}/{}".format(teff.min(), teff.max(), numpy.exp(self.grid_teff.min()), numpy.exp(self.grid_teff.max())) )
        
        # Formatting the coefficient arrays for the limb darkening calculation
        a1 = d[2]
        a1.shape = teff.size, logg.size
        a2 = d[3]
        a2.shape = teff.size, logg.size
        a3 = d[4]
        a3.shape = teff.size, logg.size
        a4 = d[5]
        a4.shape = teff.size, logg.size
        
        # Calculating the indices for the interpolation
        w_teff, ind_teff = Utils.Getaxispos_vector(teff, numpy.exp(self.grid_teff))
        w_logg, ind_logg = Utils.Getaxispos_vector(logg, self.grid_logg)
        mu = self.grid_mu
        ind_teff, ind_logg, teff = numpy.ix_(ind_teff, ind_logg, mu)
        w_teff, w_logg, teff = numpy.ix_(w_teff, w_logg, mu)
        
        # Interpolating the values
        a1 = (a1[ind_teff,ind_logg]*(1-w_teff) + a1[ind_teff+1,ind_logg]*w_teff)*(1-w_logg) + (a1[ind_teff,ind_logg+1]*(1-w_teff) + a1[ind_teff+1,ind_logg+1]*w_teff)*w_logg
        a2 = (a2[ind_teff,ind_logg]*(1-w_teff) + a2[ind_teff+1,ind_logg]*w_teff)*(1-w_logg) + (a2[ind_teff,ind_logg+1]*(1-w_teff) + a2[ind_teff+1,ind_logg+1]*w_teff)*w_logg
        a3 = (a3[ind_teff,ind_logg]*(1-w_teff) + a3[ind_teff+1,ind_logg]*w_teff)*(1-w_logg) + (a3[ind_teff,ind_logg+1]*(1-w_teff) + a3[ind_teff+1,ind_logg+1]*w_teff)*w_logg
        a4 = (a4[ind_teff,ind_logg]*(1-w_teff) + a4[ind_teff+1,ind_logg]*w_teff)*(1-w_logg) + (a4[ind_teff,ind_logg+1]*(1-w_teff) + a4[ind_teff+1,ind_logg+1]*w_teff)*w_logg
        ldc = 1 - a1*(1-mu**0.5) - a2*(1-mu) - a3*(1-mu**1.5) - a4*(1-mu**2)
        
        if (ldc < 0.).any():
            print( "Careful! There are negative limb darkening coefficients! Replacing values by 0. Below is the list (temp, logg, mu)." )
            inds_teff, inds_logg, inds_mu = (ldc < 0.).nonzero()
            print( numpy.exp(self.grid_teff[inds_teff]) )
            print( self.grid_logg[inds_logg] )
            print( self.grid_mu[inds_mu] )
            ldc[inds_teff, inds_logg, inds_mu] = 0.

        return ldc
        

######################## class Atmo_grid_BTSettl7 ########################

