# Licensed under a 3-clause BSD style license - see LICENSE

__all__ = ["Atmo_grid_spectro"]

from ..Utils.import_modules import *
from .. import Utils
from .Atmo import Atmo_grid


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
            logtemp: effective temperatures. logtemp.shape = (ntemp)
            logg: log of surface gravity. logg.shape = (nlogg)
            mu: cos(angle) of emission direction. mu.shape = (nmu)
            grid: the grid of specific intensities. grid.shape = (ntemp,nlogg,nmu)
            leff: ???
            h: ???
        wave_cut: Allows to define a lower-upper cut in wavelength [wave_low, wave_up].
        linlog (=False): If true, will rebin the data to be linear in the log space.
        
        >>> self.Flux_init()
        """
        lst = []
        for i in np.arange(len(flns)):
            # Get the log(g) and temp value from the filename
            lst.append( [i, float(flns[i].split('-')[1]), float(flns[i].split('lte')[1].split('-')[0])*100.] )
        Utils.Misc.List_sort(lst, [2,1])
        lst = np.array(lst)
        self.logtemp = np.log(list(set(lst[:,2])))
        self.logtemp.sort()
        n_temp = self.logtemp.shape[0]
        self.logg = np.array(list(set(lst[:,1])))
        self.logg.sort()
        n_logg = self.logg.shape[0]
        if n_temp*n_logg != lst.shape[0]:
            print "There is a mismatch in the number of log(g) and temp grid points"
            return
        grid = []
        mu = []
        wav = []
        for l in lst[:,0]:
            tmp = self.Flux_init_singlefile(flns[int(l)], wave_cut=wave_cut, linlog=linlog)
            grid.append(tmp[0])
            mu.append(tmp[1])
            wav.append(tmp[2])
        try:
            mu = np.array(mu)
            wav = np.array(wav)
            if mu.std(0).sum() > 1.e-6:
                print 'mu has different values'
                return
            else:
                self.mu = mu[0]
            if wav.std(0).sum() > 1.e-6:
                print 'wav has different values'
                return
            else:
                self.wav = wav[0]
        except:
            print 'mu or wav has inconsistent number of elements'
            return
        grid = np.array(grid)
        grid.shape = n_temp, n_logg, self.mu.shape[0], self.wav.shape[0]
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
        mu = np.array(lines[3].split()+lines[4].split()+lines[5].split()+lines[6].split(),dtype=float)
        # Read the info line for each grid point
        hdr = []
        grid = []
        # The first grid point is "special"
        hdr.append(lines[1].split())
        grid.append(lines[8].split()+lines[9].split()+lines[10].split()+lines[11].split())
        # Now the other lines
        for i in np.arange(12,len(lines),6):
            hdr.append(lines[i].split())
            grid.append(lines[i+2].split()+lines[i+3].split()+lines[i+4].split()+lines[i+5].split())
        hdr = np.array(hdr,dtype=float)
        # The wavelength is contained in the first column of the grid element headers.
        wav = hdr[:,0]
        grid = np.log(np.array(grid,dtype=float).T/(C*100)*wav**2)
        # There is no point in keeping grid values for mu < 0. We discard them.
        grid = grid[mu > 0.]
        mu = mu[mu > 0.]
        if wave_cut is not None:
            inds = (wav > wave_cut[0]) * (wav < wave_cut[1])
            grid = grid.take(inds, axis=-1)
            wav = wav[inds]
        if linlog:
            new_wav, self.v, self.z = Utils.Series.Resample_linlog(wav)
            ws, inds = Utils.Series.Getaxispos_vector(wav, new_wav)
            wav = new_wav
            grid = grid.take(inds, axis=-1)*(1-ws) + grid.take(inds+1, axis=-1)*ws
        return grid, mu, wav

    def Interp_orig(self, val_temp, val_logg, val_mu):
        """
        Obsolete!!!
        """
        grid = self.grid
        logtemp = self.logtemp
        logg = self.logg
        mu = self.mu
        w1temp, jtemp = self.Getaxispos(logtemp,val_temp)
        w1logg, jlogg = self.Getaxispos(logg,val_logg)
        w1mu, jmu = self.Getaxispos(mu,val_mu)
        w1temp.shape = w1temp.size,1
        w1logg.shape = w1logg.size,1
        w1mu.shape = w1mu.size,1
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
        val_mu = val_mu.reshape((val_mu.size,1))
        flux = np.exp(fl) * val_mu * self.Limb_darkening(val_mu, self.wav)
        return flux

    def Limb_darkening(self, mu, wav):
        """Limb_darkening(mu, wav)
        Returns the limb darkening factor given the cos(angle)
        of emission, mu, and the wavelength, wav, in angstroms.
        
        Note: The limb darkening law is from
            Hestroffer and Magnan, A&A, 1998, 333, 338
        """
        # We calculate the alpha power-law index, given the wavelength.
        # Lambda has to be in micrometer, hence the 1e4 factor.
        alpha = -0.023 + 0.292*(1e4/wav)
        return 1 - mu*(1-mu**alpha)

######################## class Atmo_grid_spectro ########################

