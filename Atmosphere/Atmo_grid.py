# Licensed under a 3-clause BSD style license - see LICENSE

__all__ = ["Atmo_grid"]

from ..Utils.import_modules import *
from .. import Utils


######################## class Atmo_grid ########################
class Atmo_grid:
    """Atmo_grid
    This class handles the atmosphere grid.
    """
    def __init__(self, fln, lam, dlam, flux0, ext=0.):
        """__init__
        """
        self.grid_lam = lam
        self.dlam = dlam
        self.flux0 = flux0
        self.ext = ext
        self.fln = fln
        self.Flux_init()

    def Flux_init(self):
        """Flux_init()
        Reads a band file and construct a grid.
        Calculates:
            grid_teff: effective temperatures. grid_teff.shape = (nteff)
            grid_logg: log of surface gravity. grid_logg.shape = (nlogg)
            grid_mu: cos(angle) of emission direction. grid_mu.shape = (nmu)
            grid: the grid of specific intensities. grid.shape = (nteff,nlogg,nmu)
            grid_leff: ???
            grid_h: ???
        
        >>> self.Flux_init()
        """
        f = open(self.fln,'r')
        lines = f.readlines()
        # We read the header line containing the number of temperatures (n_temp), logg (n_logg) and mu=cos(angle) (n_mu)
        n_teff, n_logg, n_mu = lines[1].split()[:3]
        n_teff = int(n_teff)
        n_logg = int(n_logg)
        n_mu = int(n_mu)
        # There should be 3 lines per grid point (teff,logg,mu): the info line and two flux lines
        # To that, we must subtract the comment line, the header line and two lines for the mu values
        if (n_teff*abs(n_logg)*3) != len(lines)-4:
            print 'It appears that the number of lines in the file is weird'
            return None
        # Read the mu values
        grid_mu = numpy.array(lines[2].split()+lines[3].split(),dtype=float)
        # Read the info line for each grid point
        hdr = []
        grid = []
        for i in numpy.arange(4,len(lines),3):
            hdr.append(lines[i].split())
            grid.append(lines[i+1].split()+lines[i+2].split())
        hdr = numpy.array(hdr,dtype=float)
        grid = numpy.log(numpy.array(grid,dtype=float)/(C*100)*self.grid_lam**2)
        hdr.shape = (n_teff,abs(n_logg),hdr.shape[1])
        grid.shape = (n_teff,abs(n_logg),n_mu)
        grid_teff = numpy.log(hdr[:,0,0])
        grid_logg = hdr[0,:,1]
        grid_leff = hdr[0,0,2]
        #jl = hdr[:,:,3]
        grid_h = hdr[:,:,4]
        #bl = hdr[:,:,5]
        #self.hdr = hdr
        self.grid = grid
        self.grid_teff = grid_teff
        self.grid_logg = grid_logg
        self.grid_mu = grid_mu
        self.grid_leff = grid_leff
        self.grid_h = grid_h
        return

    def Get_flux(self, val_logteff, val_logg, val_mu, val_area):
        """Get_flux(val_logteff, val_logg, val_mu, val_area)
        Returns the flux interpolated from the atmosphere grid.
        val_logteff: log of effective temperature
        val_logg: log of surface gravity
        val_mu: cos(angle) of angle of emission
        val_area: area of the surface element
        
        >>> self.Get_flux(val_logteff, val_logg, val_mu, val_area)
        flux
        """
        return self.Inter8(val_logteff, val_logg, val_mu, val_area)

    def Get_flux_details(self, val_logteff, val_logg, val_mu, val_area, val_v):
        """Get_flux_details(val_logteff, val_logg, val_mu, val_area, val_v)
        Returns the flux interpolated from the atmosphere grid.
        val_logteff: log of effective temperature
        val_logg: log of surface gravity
        val_mu: cos(angle) of angle of emission
        val_area: area of the surface element
        val_v: velocity of the surface element
        
        >>> self.Get_flux_details(val_logteff, val_logg, val_mu, val_area, val_v)
        flux, Keff, Teff
        """
        grid = self.grid
        logteff = self.grid_teff
        logg = self.grid_logg
        mu = self.grid_mu
        w1teff, jteff = self.Getaxispos(logteff,val_logteff)
        w1logg, jlogg = self.Getaxispos(logg,val_logg)
        w1mu, jmu = self.Getaxispos(mu,val_mu)
        flux, Keff, Teff = Utils.Inter8_photometry_details(grid, w1teff, w1logg, w1mu, jteff, jlogg, jmu, val_area, val_mu, val_v, val_logteff)
        return flux, Keff, Teff

    def Get_flux_Keff(self, val_logteff, val_logg, val_mu, val_area, val_v):
        """Get_flux_Keff(val_logteff, val_logg, val_mu, val_area, val_v)
        Returns the flux interpolated from the atmosphere grid.
        val_logteff: log of effective temperature
        val_logg: log of surface gravity
        val_mu: cos(angle) of angle of emission
        val_area: area of the surface element
        val_v: velocity of the surface element
        
        >>> self.Get_flux_Keff(val_logteff, val_logg, val_mu, val_area, val_v)
        flux, Keff
        """
        grid = self.grid
        logteff = self.grid_teff
        logg = self.grid_logg
        mu = self.grid_mu
        w1teff, jteff = self.Getaxispos(logteff,val_logteff)
        w1logg, jlogg = self.Getaxispos(logg,val_logg)
        w1mu, jmu = self.Getaxispos(mu,val_mu)
        flux, Keff = Utils.Inter8_photometry_Keff(grid, w1teff, w1logg, w1mu, jteff, jlogg, jmu, val_area, val_mu, val_v)
        return flux, Keff

    def Get_flux_nosum(self, val_logteff, val_logg, val_mu, val_area):
        """Get_flux_nosum(val_logteff, val_logg, val_mu, val_area)
        Returns the flux interpolated from the atmosphere grid.
        val_logteff: log of effective temperature
        val_logg: log of surface gravity
        val_mu: cos(angle) of angle of emission
        val_area: area of the surface element
        
        >>> self.Get_flux_nosum(val_logteff, val_logg, val_mu, val_area)
        flux
        """
        grid = self.grid
        logteff = self.grid_teff
        logg = self.grid_logg
        mu = self.grid_mu
        w1teff, jteff = self.Getaxispos(logteff,val_logteff)
        w1logg, jlogg = self.Getaxispos(logg,val_logg)
        w1mu, jmu = self.Getaxispos(mu,val_mu)
        flux = Utils.Inter8_photometry_nosum(grid, w1teff, w1logg, w1mu, jteff, jlogg, jmu, val_area, val_mu)
        return flux

    def Getaxispos(self, xx, x):
        """
        """
        if type(x) == type(numpy.array([])):
            return Utils.Getaxispos_vector(xx, x)
        else:
            return Utils.Getaxispos_scalar(xx, x)

    def Getaxispos_old(self, xx, x):
        """
        """
        ascending = xx[-1] > xx[0]
        jl = 0
        ju = xx.size
        while (ju-jl) > 1:
            jm=(ju+jl)/2
            if ascending == (x > xx[jm]):
                jl=jm
            else:
                ju=jm
        j = min(max(jl,0),xx.size-1)
        if j == xx.size-1:
            j -= 1
            #print "Reaching the end..."
        w = (x-xx[j])/(xx[j+1]-xx[j])
        return w,j

    def Inter8(self, val_teff, val_logg, val_mu, val_area):
        """
        """
#        print( "Inter8" )
        grid = self.grid
        logteff = self.grid_teff
        logg = self.grid_logg
        mu = self.grid_mu
        w1teff, jteff = self.Getaxispos(logteff,val_teff)
        w1logg, jlogg = self.Getaxispos(logg,val_logg)
        w1mu, jmu = self.Getaxispos(mu,val_mu)
        #return w1teff, jteff, w1logg, jlogg, w1mu, jmu
#        print( " Call Inter8_photometry" )
        flux = Utils.Inter8_photometry(grid, w1teff, w1logg, w1mu, jteff, jlogg, jmu, val_area, val_mu)
        return flux

    def Inter8_orig(self, val_teff, val_logg, val_mu):
        """
        Obsolete!!!
        """
        grid = self.grid
        logteff = self.grid_teff
        logg = self.grid_logg
        mu = self.grid_mu
        w1teff, jteff = self.Getaxispos(logteff,val_teff)
        w1logg, jlogg = self.Getaxispos(logg,val_logg)
        w1mu, jmu = self.Getaxispos(mu,val_mu)
        w0mu = 1.-w1mu
        w0teff = 1.-w1teff
        w0logg = 1.-w1logg
        fl = w0logg*(w0teff*(w0mu*grid[jteff,jlogg,jmu] \
                            +w1mu*grid[jteff,jlogg,jmu+1]) \
                    +w1teff*(w0mu*grid[jteff+1,jlogg,jmu] \
                            +w1mu*grid[jteff+1,jlogg,jmu+1])) \
            +w1logg*(w0teff*(w0mu*grid[jteff,jlogg+1,jmu] \
                            +w1mu*grid[jteff,jlogg+1,jmu+1]) \
                    +w1teff*(w0mu*grid[jteff+1,jlogg+1,jmu] \
                            +w1mu*grid[jteff+1,jlogg+1,jmu+1]))
        flux = numpy.exp(fl)*val_mu
        return flux

######################## class Atmo_grid ########################

