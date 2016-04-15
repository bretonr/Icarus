# Licensed under a 3-clause BSD style license - see LICENSE

__all__ = ["Atmo_grid_lithium"]

from ..Utils.import_modules import *
from .. import Utils
from .Atmo import Atmo_grid


######################## class Atmo_grid_lithium ########################
class Atmo_grid_lithium(Atmo_grid):
    """Atmo_grid_lithium
    This class handles the atmosphere grid containing a spectral
    dimension.
    """
    def __init__(self, flns, oversample=None, smooth=None, thin=None, convert=None, zp=0., wave_cut=[3000,11000], linlog=False, verbose=False, savememory=True):
        """__init__
        """
        # zp is for compatibility of spectroscopic and photometric atmosphere grids and scales the
        # zero point for the magnitude conversion.
        self.zp = zp
        self.flns = flns
        self.Flux_init(flns, oversample=oversample, smooth=smooth, thin=thin, convert=convert, wave_cut=wave_cut, linlog=linlog, verbose=verbose)
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

    def Flux_init(self, flns, oversample=None, smooth=None, thin=None, wave_cut=None, convert=None, linlog=False, verbose=False):
        """Flux_init(flns, oversample=None, smooth=None, thin=None, wave_cut=None, convert=None, linlog=False, verbose=False)
        Reads atmosphere model files and construct a grid.
        Calculates:
            logtemp: effective temperatures. logtemp.shape = (ntemp)
            logg: log of surface gravity. logg.shape = (nlogg)
            mu: cos(angle) of emission direction. mu.shape = (nmu)
            grid: the grid of specific intensities. grid.shape = (ntemp,nlogg,nmu)
            leff: ???
            h: ???

        fln: filenames
        oversample (None): Oversampling factor (integer). If provided, a cubic spline
            interpolation will be performed in order to oversample the grid in the
            wavelength dimension by a factor 'oversample'.
        smooth (None): If provided, the grid will be smoothed with a Gaussian with
            a sigma equals to 'smooth' in the wavelength dimension.
        thin (None): Thinning factor (integer). If provided, the grid will be thinned
            by keeping every other 'thin' values in the wavelength dimension.
        wave_cut (None): Allows to define a lower-upper cut in wavelength [wave_low, wave_up].
        convert (None): If not None, will append 'convert' at the end of the filename
            and save the results therein.
        linlog (False): If true, will rebin the data to be linear in the log space.
        verbose (False): verbosity.
        
        >>> self.Flux_init(flns)
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
        if 1 == 1:
            grid = []
            wav = []
            for l in lst[:,0]:
                if verbose: print( 'Reading '+flns[int(l)] )
                tmp = self.Flux_init_singlefile(flns[int(l)], oversample=oversample, smooth=smooth, thin=thin, wave_cut=wave_cut, convert=convert, linlog=linlog)
                grid.append(tmp[0])
                wav.append(tmp[1])
            if verbose: print( 'Finished reading files' )
            try:
                wav = np.array(wav)
                if wav.std(0).max() > 1.e-6:
                    print 'wav has different values'
                    return
                else:
                    wav = wav[0]
            except:
                print 'wav has inconsistent number of elements'
                return
            if verbose: print( 'Transforming grid data to array' )
            grid = np.asarray(grid)
            if verbose: print( 'Addressing the grid data shape' )
            grid.shape = n_temp, n_logg, wav.shape[0]
            self.wav = wav
            if verbose: print( 'Making the grid a class attribute' )
            self.grid = grid
        return

    def Flux_init_singlefile(self, fln, oversample=None, smooth=None, thin=None, wave_cut=None, convert=None, linlog=False):
        """Flux_init_singlefile(fln, oversample=None, smooth=None, thin=None, wave_cut=None, convert=None, linlog=False)
        Reads a band file and return the grid and wavelength.
        
        fln: filename
        oversample (None): Oversampling factor (integer). If provided, a cubic spline
            interpolation will be performed in order to oversample the grid in the
            wavelength dimension by a factor 'oversample'.
        smooth (None): If provided, the grid will be smoothed with a Gaussian with
            a sigma equals to 'smooth' in the wavelength dimension.
        thin (None): Thinning factor (integer). If provided, the grid will be thinned
            by keeping every other 'thin' values in the wavelength dimension.
        wave_cut (None): Allows to define a lower-upper cut in wavelength [wave_low, wave_up].
        convert (None): If not None, will append 'convert' at the end of the filename
            and save the results therein.
        linlog (False): If true, will rebin the data to be linear in the log space.
        
        >>> grid, wav = self.Flux_init_singlefile(fln, thin=20)
        """
        # The wavelength is contained in the first column and the flux in the second.
        wav, grid = np.loadtxt(fln, usecols=(0,1), unpack=True)
        # Grid values are log10(F_wavbda) [cgs]
        grid = 10**grid # Conversion to flux units (make sure that in the Get_flux routine does not re-correct again!)
#        grid = np.log(grid)
#        grid = np.log(grid/2.99792458e10*wav**2)
        # wavelengths are often not ordered so we re-order them
        inds = wav.argsort()
        wav = wav[inds]
        grid = grid[inds]
        if wave_cut is not None:
            inds = (wav > wave_cut[0]) * (wav < wave_cut[1])
            grid = grid[inds]
            wav = wav[inds]
        if oversample is not None:
            #interp = scipy.interpolate.interp1d(wav, grid, kind='cubic')
            interp = scipy.interpolate.UnivariateSpline(wav, grid, s=0)
            wav = np.linspace(wav[0], wav[-1], wav.size*oversample)
            grid = interp(wav)
        if smooth is not None:
            grid = scipy.ndimage.gaussian_filter1d(grid, smooth)
        if thin is not None:
            grid = grid[::thin]
            wav = wav[::thin]
        if linlog:
            new_wav, self.v, self.z = Utils.Series.Resample_linlog(wav)
            ws, inds = Utils.Series.Getaxispos_vector(wav, new_wav)
            wav = new_wav
            grid = grid.take(inds, axis=-1)*(1-ws) + grid.take(inds+1, axis=-1)*ws
        else:
            self.z0 = np.float(wav[1]/wav[0] - 1)
        if convert is not None:
            print 'Saving the data into '+fln+convert
            np.savetxt(fln+convert,np.vstack((wav,np.log10(grid))).T)
        return grid, wav

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
        val_mu.shape = val_mu.size,1
        flux = fl * val_mu
#        flux = 10**fl * val_mu
        return flux

    def Interp_orig_nomu(self, val_temp, val_logg, val_mu):
        """
        Obsolete!!!
        """
        grid = self.grid
        logtemp = self.logtemp
        logg = self.logg
        w1temp, jtemp = self.Getaxispos(logtemp,val_temp)
        w1logg, jlogg = self.Getaxispos(logg,val_logg)
        w1temp.shape = w1temp.size,1
        w1logg.shape = w1logg.size,1
        w0temp = 1.-w1temp
        w0logg = 1.-w1logg
        fl = w0logg*(w0temp*grid[jtemp,jlogg] \
                    +w1temp*grid[jtemp+1,jlogg]) \
            +w1logg*(w0temp*grid[jtemp,jlogg+1] \
                    +w1temp*grid[jtemp+1,jlogg+1])
        val_mu.shape = val_mu.size,1
        flux = fl * val_mu * Utils.Flux.Limb_darkening(self.wav/1e4, val_mu)
#        flux = 10**fl * val_mu * Limb_darkening(self.wav/1e4, val_mu)
        return flux

    def Limb_darkening(self, mu):
        """
        Returns the limb darkening for each wavelength of the grid.
        mu: cos(theta) direction of emission angle.
        
        Note: Only valid for 0.42257 < wav < 1.100 micrometer.
        From Neckel 2005.
        """
        return self.limb[0] + (self.limb[1] + (self.limb[2] + (self.limb[3] + (self.limb[4] + self.limb[5]*mu )*mu )*mu )*mu )*mu

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
        if verbose: print( 'Calculating the limb darkening grid' )
        self.mu = np.arange(0.,1.05,0.05)
        grid_mu = self.mu.copy()
        grid_mu.shape = grid_mu.size,1
        grid_mu = self.Limb_darkening(grid_mu)
        if self.savememory:
            self.grid_mu = grid_mu
        else:
            g = np.array([self.grid * m for m in mu])
            self.grid = np.ascontiguousarray(g.swapaxes(0,1).swapaxes(1,2))

######################## class Atmo_grid_lithium ########################


