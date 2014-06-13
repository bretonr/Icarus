# Licensed under a 3-clause BSD style license - see LICENSE

__all__ = ["Atmo_grid"]

from ..Utils.import_modules import *
from .. import Utils


######################## class Atmo_grid ########################
class Atmo_grid:
    """Atmo_grid
    This class handles the atmosphere grid.
    """
    def __init__(self, fln, wav, dwav, flux0, ext=0.):
        """__init__
        """
        self.wav = wav
        self.dwav = dwav
        self.flux0 = flux0
        self.ext = ext
        self.fln = fln
        self.Flux_init()

    def Flux_init(self):
        """Flux_init()
        Reads a band file and construct a grid.
        Calculates:
            temp: effective temperatures. temp.shape = (ntemp)
            logg: log of surface gravity. logg.shape = (nlogg)
            mu: cos(angle) of emission direction. mu.shape = (nmu)
            grid: the grid of specific intensities. grid.shape = (ntemp,nlogg,nmu)
            leff: ???
            h: ???
        
        >>> self.Flux_init()
        """
        f = open(self.fln,'r')
        lines = f.readlines()
        # We read the header line containing the number of temperatures (n_temp), logg (n_logg) and mu=cos(angle) (n_mu)
        n_temp, n_logg, n_mu = lines[1].split()[:3]
        n_temp = int(n_temp)
        n_logg = int(n_logg)
        n_mu = int(n_mu)
        # There should be 3 lines per grid point (temp,logg,mu): the info line and two flux lines
        # To that, we must subtract the comment line, the header line and two lines for the mu values
        if (n_temp*abs(n_logg)*3) != len(lines)-4:
            print 'It appears that the number of lines in the file is weird'
            return None
        # Read the mu values
        mu = numpy.array(lines[2].split()+lines[3].split(),dtype=float)
        # Read the info line for each grid point
        hdr = []
        grid = []
        for i in numpy.arange(4,len(lines),3):
            hdr.append(lines[i].split())
            grid.append(lines[i+1].split()+lines[i+2].split())
        hdr = numpy.array(hdr,dtype=float)
        grid = numpy.log(numpy.array(grid,dtype=float)/(C*100)*self.wav**2)
        hdr.shape = (n_temp,abs(n_logg),hdr.shape[1])
        grid.shape = (n_temp,abs(n_logg),n_mu)
        logtemp = numpy.log(hdr[:,0,0])
        logg = hdr[0,:,1]
        leff = hdr[0,0,2]
        #jl = hdr[:,:,3]
        h = hdr[:,:,4]
        #bl = hdr[:,:,5]
        #self.hdr = hdr
        self.grid = grid
        self.logtemp = logtemp
        self.logg = logg
        self.mu = mu
        self.leff = leff
        self.h = h
        return

    def Get_flux(self, val_logtemp, val_logg, val_mu, val_area):
        """Get_flux(val_logtemp, val_logg, val_mu, val_area)
        Returns the flux interpolated from the atmosphere grid.
        val_logtemp: log of effective temperature
        val_logg: log of surface gravity
        val_mu: cos(angle) of angle of emission
        val_area: area of the surface element
        
        >>> self.Get_flux(val_logtemp, val_logg, val_mu, val_area)
        flux
        """
        grid = self.grid
        logtemp = self.logtemp
        logg = self.logg
        mu = self.mu
        w1temp, jtemp = self.Getaxispos(logtemp,val_logtemp)
        w1logg, jlogg = self.Getaxispos(logg,val_logg)
        w1mu, jmu = self.Getaxispos(mu,val_mu)
        flux = Utils.Grid.Inter8_photometry(grid, w1temp, w1logg, w1mu, jtemp, jlogg, jmu, val_area, val_mu)
        return flux

    def Get_flux_details(self, val_logtemp, val_logg, val_mu, val_area, val_v):
        """Get_flux_details(val_logtemp, val_logg, val_mu, val_area, val_v)
        Returns the flux interpolated from the atmosphere grid.
        val_logtemp: log of effective temperature
        val_logg: log of surface gravity
        val_mu: cos(angle) of angle of emission
        val_area: area of the surface element
        val_v: velocity of the surface element
        
        >>> self.Get_flux_details(val_logtemp, val_logg, val_mu, val_area, val_v)
        flux, Keff, temp
        """
        grid = self.grid
        logtemp = self.logtemp
        logg = self.logg
        mu = self.mu
        w1temp, jtemp = self.Getaxispos(logtemp,val_logtemp)
        w1logg, jlogg = self.Getaxispos(logg,val_logg)
        w1mu, jmu = self.Getaxispos(mu,val_mu)
        flux, Keff, temp = Utils.Grid.Inter8_photometry_details(grid, w1temp, w1logg, w1mu, jtemp, jlogg, jmu, val_area, val_mu, val_v, val_logtemp)
        return flux, Keff, temp

    def Get_flux_Keff(self, val_logtemp, val_logg, val_mu, val_area, val_v):
        """Get_flux_Keff(val_logtemp, val_logg, val_mu, val_area, val_v)
        Returns the flux interpolated from the atmosphere grid.
        val_logtemp: log of effective temperature
        val_logg: log of surface gravity
        val_mu: cos(angle) of angle of emission
        val_area: area of the surface element
        val_v: velocity of the surface element
        
        >>> self.Get_flux_Keff(val_logtemp, val_logg, val_mu, val_area, val_v)
        flux, Keff
        """
        grid = self.grid
        logtemp = self.logtemp
        logg = self.logg
        mu = self.mu
        w1temp, jtemp = self.Getaxispos(logtemp,val_logtemp)
        w1logg, jlogg = self.Getaxispos(logg,val_logg)
        w1mu, jmu = self.Getaxispos(mu,val_mu)
        flux, Keff = Utils.Grid.Inter8_photometry_Keff(grid, w1temp, w1logg, w1mu, jtemp, jlogg, jmu, val_area, val_mu, val_v)
        return flux, Keff

    def Get_flux_nosum(self, val_logtemp, val_logg, val_mu, val_area):
        """Get_flux_nosum(val_logtemp, val_logg, val_mu, val_area)
        Returns the flux interpolated from the atmosphere grid.
        val_logtemp: log of effective temperature
        val_logg: log of surface gravity
        val_mu: cos(angle) of angle of emission
        val_area: area of the surface element
        
        >>> self.Get_flux_nosum(val_logtemp, val_logg, val_mu, val_area)
        flux
        """
        grid = self.grid
        logtemp = self.logtemp
        logg = self.logg
        mu = self.mu
        w1temp, jtemp = self.Getaxispos(logtemp,val_logtemp)
        w1logg, jlogg = self.Getaxispos(logg,val_logg)
        w1mu, jmu = self.Getaxispos(mu,val_mu)
        flux = Utils.Grid.Inter8_photometry_nosum(grid, w1temp, w1logg, w1mu, jtemp, jlogg, jmu, val_area, val_mu)
        return flux

    def Getaxispos(self, xx, x):
        """
        """
        if isinstance(x, (list, tuple, numpy.ndarray)):
            return Utils.Series.Getaxispos_vector(xx, x)
        else:
            return Utils.Series.Getaxispos_scalar(xx, x)

    def Getaxispos_old(self, xx, x):
        """
        OBSOLETE!
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

    def Inter8_orig(self, val_logtemp, val_logg, val_mu):
        """
        Obsolete!!!
        """
        grid = self.grid
        logtemp = self.logtemp
        logg = self.logg
        mu = self.mu
        w1temp, jtemp = self.Getaxispos(logtemp,val_logtemp)
        w1logg, jlogg = self.Getaxispos(logg,val_logg)
        w1mu, jmu = self.Getaxispos(mu,val_mu)
        w0mu = 1.-w1mu
        w0temp = 1.-w1temp
        w0logg = 1.-w1logg
        fl = w0logg*(w0temp*(w0mu*grid[jtemp,jlogg,jmu] \
                            +w1mu*grid[jtemp,jlogg,jmu+1]) \
                    +w1temp*(w0mu*grid[jtemp+1,jlogg,jmu] \
                            +w1mu*grid[jtemp+1,jlogg,jmu+1])) \
            +w1logg*(w0temp*(w0mu*grid[jtemp,jlogg+1,jmu] \
                            +w1mu*grid[jtemp,jlogg+1,jmu+1]) \
                    +w1temp*(w0mu*grid[jtemp+1,jlogg+1,jmu] \
                            +w1mu*grid[jtemp+1,jlogg+1,jmu+1]))
        flux = numpy.exp(fl)*val_mu
        return flux

######################## class Atmo_grid ########################

