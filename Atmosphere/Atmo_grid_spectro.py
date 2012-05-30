# Licensed under a 3-clause BSD style license - see LICENSE

__all__ = ["Atmo_grid_spectro"]

from ..Utils.import_modules import *
from .. import Utils
from .Atmo_grid import Atmo_grid


######################## class Atmo_grid_spectro ########################
class Atmo_grid_spectro(Atmo_grid):
    """Atmo_grid_spectro
    This class handles the atmosphere grid containing a spectral
    dimension.
    """
    def __init__(self, flns, wave_cut=[3000,11000], linlog=False):
        """__init__
        """
        self.flns = flns
        self.Flux_init(flns, wave_cut=wave_cut, linlog=linlog)

    def Flux_init(self, flns, wave_cut=None, linlog=False):
        """Flux_init(flns, wave_cut=None, linlog=False)
        Reads a band file and construct a grid.
        Calculates:
            grid_teff: effective temperatures. grid_teff.shape = (nteff)
            grid_logg: log of surface gravity. grid_logg.shape = (nlogg)
            grid_mu: cos(angle) of emission direction. grid_mu.shape = (nmu)
            grid: the grid of specific intensities. grid.shape = (nteff,nlogg,nmu)
            grid_leff: ???
            grid_h: ???
        wave_cut: Allows to define a lower-upper cut in wavelength [wave_low, wave_up].
        linlog (=False): If true, will rebin the data to be linear in the log space.
        
        >>> self.Flux_init()
        """
        lst = []
        for i in numpy.arange(len(flns)):
            # Get the log(g) and teff value from the filename
            lst.append( [i, float(flns[i].split('-')[1]), float(flns[i].split('lte')[1].split('-')[0])*100.] )
        bretonr_utils.List_sort(lst, [2,1])
        lst = numpy.array(lst)
        #self.grid_teff = numpy.array(list(set(lst[:,2])))
        self.grid_teff = numpy.log(list(set(lst[:,2])))
        self.grid_teff.sort()
        n_teff = self.grid_teff.shape[0]
        self.grid_logg = numpy.array(list(set(lst[:,1])))
        self.grid_logg.sort()
        n_logg = self.grid_logg.shape[0]
        if n_teff*n_logg != lst.shape[0]:
            print "There is a mismatch in the number of log(g) and teff grid points"
            return
        grid = []
        grid_mu = []
        grid_lam = []
        for l in lst[:,0]:
            tmp = self.Flux_init_singlefile(flns[int(l)], wave_cut=wave_cut, linlog=linlog)
            grid.append(tmp[0])
            grid_mu.append(tmp[1])
            grid_lam.append(tmp[2])
        try:
            grid_mu = numpy.array(grid_mu)
            grid_lam = numpy.array(grid_lam)
            if grid_mu.std(0).sum() > 1.e-6:
                print 'grid_mu has different values'
                return
            else:
                self.grid_mu = grid_mu[0]
            if grid_lam.std(0).sum() > 1.e-6:
                print 'grid_lam has different values'
                return
            else:
                self.grid_lam = grid_lam[0]
        except:
            print 'grid_mu or grid_lam has inconsistent number of elements'
            return
        grid = numpy.array(grid)
        grid.shape = n_teff, n_logg, self.grid_mu.shape[0], self.grid_lam.shape[0]
        self.grid = grid
        return

    def Flux_init_singlefile(self, fln, wave_cut=None, linlog=False):
        """Flux_init_singlefile(fln, linlog=False)
        Reads a band file and construct a grid.
        
        wave_cut: Allows to define a lower-upper cut in wavelength [wave_low, wave_up].
        linlog (=False): If true, will rebin the data to be linear in the log space.
        >>>
        """
        f = open(fln,'r')
        lines = f.read()
        lines = lines.replace('D+','E+')
        lines = lines.replace('D-','E-')
        lines = lines.splitlines()
        # Read the mu values
        grid_mu = numpy.array(lines[3].split()+lines[4].split()+lines[5].split()+lines[6].split(),dtype=float)
        # Read the info line for each grid point
        hdr = []
        grid = []
        # The first grid point is "special"
        hdr.append(lines[1].split())
        grid.append(lines[8].split()+lines[9].split()+lines[10].split()+lines[11].split())
        # Now the other lines
        for i in numpy.arange(12,len(lines),6):
            hdr.append(lines[i].split())
            grid.append(lines[i+2].split()+lines[i+3].split()+lines[i+4].split()+lines[i+5].split())
        hdr = numpy.array(hdr,dtype=float)
        # The wavelength is contained in the first column of the grid element headers.
        grid_lam = hdr[:,0]
        grid = numpy.log(numpy.array(grid,dtype=float).T/(C*100)*grid_lam**2)
#        grid = numpy.array(grid,dtype=float).T
        # There is no point in keeping grid values for mu < 0. We discard them.
        grid = grid[grid_mu > 0.]
        grid_mu = grid_mu[grid_mu > 0.]
        if wave_cut is not None:
            inds = (grid_lam > wave_cut[0]) * (grid_lam < wave_cut[1])
            grid = grid.take(inds, axis=-1)
            grid_lam = grid_lam[inds]
        if linlog:
            new_grid_lam, self.v, self.z = Utils.Resample_linlog(grid_lam)
            ws, inds = Utils.Getaxispos_vector(grid_lam, new_grid_lam)
            grid_lam = new_grid_lam
            grid = grid.take(inds, axis=-1)*(1-ws) + grid.take(inds+1, axis=-1)*ws
        return grid, grid_mu, grid_lam

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
        w1teff.shape = w1teff.size,1
        w1logg.shape = w1logg.size,1
        w1mu.shape = w1mu.size,1
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
        val_mu = val_mu.reshape((val_mu.size,1))
        flux = numpy.exp(fl) * val_mu * self.Limb_darkening(val_mu, self.grid_lam)
        #return jmu,jteff,jlogg,grid[jteff,jlogg,jmu],fl,flux
        return flux

    def Limb_darkening(self, mu, lam):
        """Limb_darkening(mu, lam)
        Returns the limb darkening factor given the cos(angle)
        of emission, mu, and the wavelength, lam, in angstroms.
        
        Note: The limb darkening law is from
            Hestroffer and Magnan, A&A, 1998, 333, 338
        """
        # We calculate the alpha power-law index, given the wavelength.
        # Lambda has to be in micrometer, hence the 1e4 factor.
        alpha = -0.023 + 0.292*(1e4/lam)
        return 1 - mu*(1-mu**alpha)

######################## class Atmo_grid_spectro ########################

