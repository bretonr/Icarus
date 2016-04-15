# Licensed under a 3-clause BSD style license - see LICENSE

__all__ = ["Atmo_grid_BTSettl7"]

from ..Utils.import_modules import *
from .. import Utils
from .Atmo import Atmo_grid


######################## class Atmo_grid_BTSettl7 ########################
class Atmo_phot_BTSettl7(Atmo_grid):
    """Atmo_phot_BTSettl7(Atmo_grid)
    This class handles the atmosphere grid for photometric data
    from the BT-Settl.7 atmosphere models.
    """
    def __init__(self, fln, wav, dwav, zp, ext=0., logg_lims=[3.,5.], AB=True):
        """__init__(self, fln, wav, dwav, zp, ext=0., logg_lims=[3.,5.], AB=True)
        """
        self.wav = wav
        self.dwav = dwav
        self.zp = zp
        self.ext = ext
        self.fln = fln
        self.meta = {'zp':zp, 'ext':ext}
        self.Flux_init(logg_lims=logg_lims, AB=AB)

    def Flux_init(self, logg_lims=[3.,5.], AB=True):
        """Flux_init(logg_lims=[3.,5.], AB=True)
        Reads a band file and construct a grid.
        
        The data files are: temp, logg, flux.
        
        logg_lims ([3.,5.]): provides the min and max value of logg.
        AB (True): If the grid is in the AB system, only a multiplication by PI needs to be done.
            Otherwise, needs to multiply by c*100/lambda**2.
        
        Calculates:
            temp: effective temperatures. temp.shape = (ntemp)
            logg: log of surface gravity. logg.shape = (nlogg)
            mu: cos(angle) of emission direction. mu.shape = (nmu)
            grid: the grid of specific intensities. grid.shape = (ntemp,nlogg,nmu)
        
        >>> self.Flux_init(logg_lims=[3.,5.])
        """
        # Load the data
        temp,logg,flux = np.loadtxt(self.fln, unpack=True)
        # Extract the unique values of logg
        grid_logg = np.unique(logg)
        grid_logg = grid_logg[(grid_logg>=logg_lims[0])*(grid_logg<=logg_lims[1])]
        n_logg = grid_logg.size
        
        # Trim the atmosphere grid to keep only the elements within the logg subset
        inds = np.zeros(temp.size, dtype=bool)
        for i in xrange(grid_logg.size):
            inds += logg == grid_logg[i]
        
        temp = temp[inds]
        logg = logg[inds]
        flux = flux[inds]
        
        # Trim the atmosphere grid to keep only the elements within the temp subset
        inds = np.zeros(temp.size, dtype=bool)
        grid_temp = []
        for t in np.unique(temp):
            if (temp == t).sum() == n_logg:
                grid_temp.append( t )
                inds += temp == t
        
        grid_temp = np.array(grid_temp)
        n_temp = grid_temp.size
        temp = temp[inds]
        logg = logg[inds]
        flux = flux[inds]
        
        if temp.size != n_temp*n_logg:
            print( "The number of entries does not constitute a proper grid! (n_total, n_temp, n_logg)", temp.size, n_temp, n_logg )
        
        # Sorting the grid
        inds = np.lexsort((logg,temp))
        
        temp.shape = n_temp, n_logg
        logg.shape = n_temp, n_logg
        flux.shape = n_temp, n_logg
        
        # Creating the mu values
        n_mu = 16
        grid_mu = np.linspace(0., 1., 16)
        
        # Making class variables
        self.logtemp = np.log(grid_temp)
        self.logg = grid_logg.copy()
        self.mu = grid_mu.copy()
        
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
                mu_factor = Utils.Flux.Limb_darkening(np.r_[self.wav]*1e4, mu)[:,0]
                #mu_factor = np.ones_like(mu_factor)
        else:
            mu = grid_mu.copy()
            mu.shape = n_mu,1
            mu_factor = Utils.Flux.Limb_darkening(np.r_[self.wav]*1e4, mu)[:,0]
        
        # Applying the limb darkening
        grid = np.resize(flux, (n_mu, n_temp, n_logg))
        grid = grid.swapaxes(0, 1).swapaxes(1, 2) * mu_factor
        if AB:
            grid = np.log(grid*4/cts.PI**2)
        else:
            grid = np.log(grid*4/cts.PI**2 / (cts.c*1e10) * (self.wav*1e8)**2)
        
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
        d = np.loadtxt(fln, usecols=(0,1,4,5,6,7), unpack=True)
        
        # Determining the range of logg and temperature, making sure everything is fine.
        logg = np.unique(d[0])
        if d[0].size%logg.size != 0:
            raise RuntimeError( "The number of logg values is odd and does not match the total number of entries." )
        temp = np.unique(d[1])
        if d[1].size%temp.size != 0:
            raise RuntimeError( "The number of temp values is odd and does not match the total number of entries." )
        if d[0].size != logg.size*temp.size:
            raise RuntimeError( "The total number of entries should equal the number of logg times the number of temp." )
        if logg.min() > self.logg.min() or logg.max() < self.logg.max():
            raise RuntimeError( "Careful! The min/max of the limb darkening are {}/{}, versus those of the atmosphere grid {}/{}".format(logg.min(), logg.max(), self.logg.min(), self.logg.max()) )
        if temp.min() > np.exp(self.logtemp.min()) or temp.max() < np.exp(self.logtemp.max()):
            raise RuntimeError( "Careful! The min/max of the limb darkening are {}/{}, versus those of the atmosphere grid {}/{}".format(temp.min(), temp.max(), np.exp(self.logtemp.min()), np.exp(self.logtemp.max())) )
        
        # Formatting the coefficient arrays for the limb darkening calculation
        a1 = d[2]
        a1.shape = temp.size, logg.size
        a2 = d[3]
        a2.shape = temp.size, logg.size
        a3 = d[4]
        a3.shape = temp.size, logg.size
        a4 = d[5]
        a4.shape = temp.size, logg.size
        
        # Calculating the indices for the interpolation
        w_temp, ind_temp = Utils.Getaxispos_vector(temp, np.exp(self.logtemp))
        w_logg, ind_logg = Utils.Getaxispos_vector(logg, self.logg)
        mu = self.mu
        ind_temp, ind_logg, temp = np.ix_(ind_temp, ind_logg, mu)
        w_temp, w_logg, temp = np.ix_(w_temp, w_logg, mu)
        
        # Interpolating the values
        a1 = (a1[ind_temp,ind_logg]*(1-w_temp) + a1[ind_temp+1,ind_logg]*w_temp)*(1-w_logg) + (a1[ind_temp,ind_logg+1]*(1-w_temp) + a1[ind_temp+1,ind_logg+1]*w_temp)*w_logg
        a2 = (a2[ind_temp,ind_logg]*(1-w_temp) + a2[ind_temp+1,ind_logg]*w_temp)*(1-w_logg) + (a2[ind_temp,ind_logg+1]*(1-w_temp) + a2[ind_temp+1,ind_logg+1]*w_temp)*w_logg
        a3 = (a3[ind_temp,ind_logg]*(1-w_temp) + a3[ind_temp+1,ind_logg]*w_temp)*(1-w_logg) + (a3[ind_temp,ind_logg+1]*(1-w_temp) + a3[ind_temp+1,ind_logg+1]*w_temp)*w_logg
        a4 = (a4[ind_temp,ind_logg]*(1-w_temp) + a4[ind_temp+1,ind_logg]*w_temp)*(1-w_logg) + (a4[ind_temp,ind_logg+1]*(1-w_temp) + a4[ind_temp+1,ind_logg+1]*w_temp)*w_logg
        ldc = 1 - a1*(1-mu**0.5) - a2*(1-mu) - a3*(1-mu**1.5) - a4*(1-mu**2)
        
        if (ldc < 0.).any():
            print( "Careful! There are negative limb darkening coefficients! Replacing values by 0. Below is the list (temp, logg, mu)." )
            inds_temp, inds_logg, inds_mu = (ldc < 0.).nonzero()
            print( np.exp(self.logtemp[inds_temp]) )
            print( self.logg[inds_logg] )
            print( self.mu[inds_mu] )
            ldc[inds_temp, inds_logg, inds_mu] = 0.

        return ldc
        

######################## class Atmo_grid_BTSettl7 ########################

