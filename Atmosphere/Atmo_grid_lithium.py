# Licensed under a 3-clause BSD style license - see LICENSE

__all__ = ["Atmo_grid_lithium"]

from ..Utils.import_modules import *
from .. import Utils
from .Atmo_grid import Atmo_grid


######################## class Atmo_grid_lithium ########################
class Atmo_grid_lithium(Atmo_grid):
    """Atmo_grid_lithium
    This class handles the atmosphere grid containing a spectral
    dimension.
    """
    def __init__(self, flns, oversample=None, smooth=None, thin=None, convert=None, flux0=1, wave_cut=[3000,11000], linlog=False, verbose=False, savememory=True):
        """__init__
        """
        # flux0 is for compatibility of spectroscopic and photometric atmosphere grids and scales the
        # zero point for the magnitude conversion.
        self.flux0 = flux0
        self.flns = flns
        self.Flux_init(flns, oversample=oversample, smooth=smooth, thin=thin, convert=convert, wave_cut=wave_cut, linlog=linlog, verbose=verbose)
        self.Coeff_limb_darkening(self.grid_lam/1e4, verbose=verbose)
        self.Make_limb_grid(verbose=verbose, savememory=savememory)

    def Coeff_limb_darkening(self, lam, verbose=False):
        """
        Calculates the limb darkening coefficients.
        lam: wavelength in micrometer.
        verbose (=False): verbosity.
        
        Note: Only valid for 0.42257 < lam < 1.100 micrometer.
        From Neckel 2005.
        """
        if verbose: print( "Calculating limb darkening coefficients" )
        def L_422_1100(lam):
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
            lam5 = lam**5
            a_0 = a_00 + a_01/lam
            a_1 = a_10 + a_11/lam + a_15/lam5
            a_2 = a_20 + a_25/lam5
            a_3 = a_30 + a_35/lam5
            a_4 = a_40 + a_45/lam5
            a_5 = a_50 + a_55/lam5
            return a_0, a_1, a_2, a_3, a_4, a_5
        def L_385_422(lam):
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
            lam5 = lam**5
            a_0 = a_00 + a_01/lam
            a_1 = a_10 + a_11/lam + a_15/lam5
            a_2 = a_20 + a_25/lam5
            a_3 = a_30 + a_35/lam5
            a_4 = a_40 + a_45/lam5
            a_5 = a_50 + a_55/lam5
            return a_0, a_1, a_2, a_3, a_4, a_5
        def L_300_372(lam):
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
            lam5 = lam**5
            a_0 = a_00 + a_01/lam
            a_1 = a_10 + a_11/lam + a_15/lam5
            a_2 = a_20 + a_25/lam5
            a_3 = a_30 + a_35/lam5
            a_4 = a_40 + a_45/lam5
            a_5 = a_50 + a_55/lam5
            return a_0, a_1, a_2, a_3, a_4, a_5
        self.limb = numpy.empty((6,lam.shape[0]))
        inds = lam<0.37298
        self.limb[:,inds] = L_300_372(lam[inds])
        inds = (lam<0.42257)*(lam>0.37298)
        self.limb[:,inds] = L_385_422(lam[inds])
        inds = lam>0.42257
        self.limb[:,inds] = L_422_1100(lam[inds])
        return

    def Flux_init(self, flns, oversample=None, smooth=None, thin=None, wave_cut=None, convert=None, linlog=False, verbose=False):
        """Flux_init(flns, oversample=None, smooth=None, thin=None, wave_cut=None, convert=None, linlog=False, verbose=False)
        Reads atmosphere model files and construct a grid.
        Calculates:
            grid_teff: effective temperatures. grid_teff.shape = (nteff)
            grid_logg: log of surface gravity. grid_logg.shape = (nlogg)
            grid_mu: cos(angle) of emission direction. grid_mu.shape = (nmu)
            grid: the grid of specific intensities. grid.shape = (nteff,nlogg,nmu)
            grid_leff: ???
            grid_h: ???

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
        for i in numpy.arange(len(flns)):
            # Get the log(g) and teff value from the filename
            lst.append( [i, float(flns[i].split('-')[1]), float(flns[i].split('lte')[1].split('-')[0])*100.] )
        bretonr_utils.List_sort(lst, [2,1])
        lst = numpy.array(lst)
        self.grid_teff = numpy.log(list(set(lst[:,2])))
        self.grid_teff.sort()
        n_teff = self.grid_teff.shape[0]
        self.grid_logg = numpy.array(list(set(lst[:,1])))
        self.grid_logg.sort()
        n_logg = self.grid_logg.shape[0]
        if n_teff*n_logg != lst.shape[0]:
            print "There is a mismatch in the number of log(g) and teff grid points"
            return
        if 1 == 1:
            grid = []
            grid_lam = []
            for l in lst[:,0]:
                if verbose: print( 'Reading '+flns[int(l)] )
                tmp = self.Flux_init_singlefile(flns[int(l)], oversample=oversample, smooth=smooth, thin=thin, wave_cut=wave_cut, convert=convert, linlog=linlog)
                grid.append(tmp[0])
                grid_lam.append(tmp[1])
            if verbose: print( 'Finished reading files' )
            try:
                grid_lam = numpy.array(grid_lam)
                if grid_lam.std(0).max() > 1.e-6:
                    print 'grid_lam has different values'
                    return
                else:
                    grid_lam = grid_lam[0]
            except:
                print 'grid_lam has inconsistent number of elements'
                return
            if verbose: print( 'Transforming grid data to array' )
            grid = numpy.asarray(grid)
            if verbose: print( 'Addressing the grid data shape' )
            grid.shape = n_teff, n_logg, grid_lam.shape[0]
            self.grid_lam = grid_lam
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
        
        >>> grid, grid_lam = self.Flux_init_singlefile(fln, thin=20)
        """
        # The wavelength is contained in the first column and the flux in the second.
        grid_lam, grid = numpy.loadtxt(fln, usecols=(0,1), unpack=True)
        # Grid values are log10(F_lambda) [cgs]
        grid = 10**grid # Conversion to flux units (make sure that in the Get_flux routine does not re-correct again!)
#        grid = numpy.log(grid)
#        grid = numpy.log(grid/2.99792458e10*grid_lam**2)
        # wavelengths are often not ordered so we re-order them
        inds = grid_lam.argsort()
        grid_lam = grid_lam[inds]
        grid = grid[inds]
        if wave_cut is not None:
            inds = (grid_lam > wave_cut[0]) * (grid_lam < wave_cut[1])
            grid = grid[inds]
            grid_lam = grid_lam[inds]
        if oversample is not None:
            #interp = scipy.interpolate.interp1d(grid_lam, grid, kind='cubic')
            interp = scipy.interpolate.UnivariateSpline(grid_lam, grid, s=0)
            grid_lam = numpy.linspace(grid_lam[0], grid_lam[-1], grid_lam.size*oversample)
            grid = interp(grid_lam)
        if smooth is not None:
            grid = scipy.ndimage.gaussian_filter1d(grid, smooth)
        if thin is not None:
            grid = grid[::thin]
            grid_lam = grid_lam[::thin]
        if linlog:
            new_grid_lam, self.v, self.z = Utils.Resample_linlog(grid_lam)
            ws, inds = Utils.Getaxispos_vector(grid_lam, new_grid_lam)
            grid_lam = new_grid_lam
            grid = grid.take(inds, axis=-1)*(1-ws) + grid.take(inds+1, axis=-1)*ws
        else:
            self.z0 = numpy.float(grid_lam[1]/grid_lam[0] - 1)
        if convert is not None:
            print 'Saving the data into '+fln+convert
            numpy.savetxt(fln+convert,numpy.vstack((grid_lam,numpy.log10(grid))).T)
        return grid, grid_lam

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
        val_mu.shape = val_mu.size,1
        flux = fl * val_mu
#        flux = 10**fl * val_mu
        return flux

    def Inter8_orig_nomu(self, val_teff, val_logg, val_mu):
        """
        Obsolete!!!
        """
        grid = self.grid
        logteff = self.grid_teff
        logg = self.grid_logg
        w1teff, jteff = self.Getaxispos(logteff,val_teff)
        w1logg, jlogg = self.Getaxispos(logg,val_logg)
        w1teff.shape = w1teff.size,1
        w1logg.shape = w1logg.size,1
        w0teff = 1.-w1teff
        w0logg = 1.-w1logg
        fl = w0logg*(w0teff*grid[jteff,jlogg] \
                    +w1teff*grid[jteff+1,jlogg]) \
            +w1logg*(w0teff*grid[jteff,jlogg+1] \
                    +w1teff*grid[jteff+1,jlogg+1])
        val_mu.shape = val_mu.size,1
        flux = fl * val_mu * Utils.Limb_darkening(self.grid_lam/1e4, val_mu)
#        flux = 10**fl * val_mu * Limb_darkening(self.grid_lam/1e4, val_mu)
        return flux

    def Limb_darkening(self, mu):
        """
        Returns the limb darkening for each wavelength of the grid.
        mu: cos(theta) direction of emission angle.
        
        Note: Only valid for 0.42257 < lam < 1.100 micrometer.
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
            in the modified Inter8 function.
        verbose (=False): verbosity.
        """
        self.savememory = savememory
        if verbose: print( 'Calculating the limb darkening grid' )
        self.grid_mu = numpy.arange(0.,1.05,0.05)
        mu = self.grid_mu.copy()
        mu.shape = mu.size,1
        mu = self.Limb_darkening(mu)
        if self.savememory:
            self.mu = mu
        else:
#        print( mu.shape )
#        g = numpy.array([self.grid * m for m in mu])
#        print( g.shape )
#        g = g.swapaxes(0,1).swapaxes(1,2)
#        print( g.shape )
#        self.grid = g.copy()
            g = numpy.array([self.grid * m for m in mu])
            self.grid = numpy.ascontiguousarray(g.swapaxes(0,1).swapaxes(1,2))

######################## class Atmo_grid_lithium ########################


######################## class Atmo_grid_lithium_doppler ########################
class Atmo_grid_lithium_doppler(Atmo_grid_lithium):
    """
    This class inherits from Atmo_grid_lithium. The difference is
    that the wavelengths are resampled to be on a linear spacing in
    the log of the wavelength, which makes it linear in velocity
    shifts. The flux calculation takes an extra parameter for the
    velocity
    """
    def __init__(self, flns, oversample=None, smooth=None, thin=None, convert=None, flux0=1, wave_cut=[3000,11000], verbose=False, savememory=True):
        """
        """
        if verbose: print( 'Reading atmosphere grid files' )
        #Atmo_grid_lithium.__init__(self, flns, oversample=oversample, smooth=smooth, thin=thin, convert=convert, flux0=flux0, wave_cut=wave_cut, linlog=True, verbose=verbose, savememory=savememory)
        Atmo_grid_lithium.__init__(self, flns, oversample=oversample, smooth=smooth, thin=thin, convert=convert, flux0=flux0, wave_cut=wave_cut, linlog=False, verbose=verbose, savememory=savememory)
        #print( 'Rebinning to linear in logarithmic spacing' )
        #self.__Make_log()
    
    def Get_flux_doppler(self, val_logteff, val_logg, val_mu, val_vel, val_area):
        """Get_flux_doppler(val_logteff, val_logg, val_mu, val_vel, val_area)
        Returns the flux interpolated from the atmosphere grid.
        val_logteff: log of effective temperature
        val_logg: log of surface gravity
        val_mu: cos(angle) of angle of emission
        val_vel: velocity of the grid point in units of speed of light
        val_area: area of the surface element
        
        >>> flux = self.Get_flux_doppler(val_logteff, val_logg, val_mu, val_vel, val_area)
        """
        grid = self.grid
        logteff = self.grid_teff
        logg = self.grid_logg
        mu = self.grid_mu
        wteff, jteff = self.Getaxispos(logteff,val_logteff)
        wlogg, jlogg = self.Getaxispos(logg,val_logg)
        wmu, jmu = self.Getaxispos(mu,val_mu)
        # Here we convert val_vel to be the number of bins which
        # corresponds to the Doppler shift. That is z_obs / z_gridsampling.
        # This is more precise than just v_obs / v_gridsampling.
        #val_vel = (numpy.sqrt((1+val_vel)/(1-val_vel)) - 1) / self.z
        #val_vel /= self.v
        #jlam = numpy.floor(val_vel).astype('i')
        #wlam = val_vel - jlam
        if self.savememory:
            mu_grid = self.mu
            #flux = Utils.Inter8_doppler_savememory(grid, wteff, wlogg, wmu, wlam, jteff, jlogg, jmu, jlam, mu_grid, val_area, val_mu)
            if self.z0:
                flux = Utils.Inter8_doppler_savememory_linear(grid, wteff, wlogg, wmu, jteff, jlogg, jmu, mu_grid, val_area, val_mu, val_vel, self.z0)
            else:
                print( 'Hey! Wake up! The grid is not linear in lambda and has been transformed to linear in log(lambda)!' )
        else:
            flux = Utils.Inter8_doppler(grid, wteff, wlogg, wmu, wlam, jteff, jlogg, jmu, jlam, val_area, val_mu)
        return flux
    
    def Get_flux_doppler_nomu(self, val_logteff, val_logg, val_mu, val_vel, val_area):
        """Get_flux_doppler_nomu(val_logteff, val_logg, val_mu, val_vel, val_area)
        Returns the flux interpolated from the atmosphere grid.
        val_logteff: log of effective temperature
        val_logg: log of surface gravity
        val_mu: cos(angle) of angle of emission
        val_vel: velocity of the grid point in units of speed of light
        val_area: area of the surface element
        
        >>> flux = self.Get_flux_doppler_nomu(val_logteff, val_logg, val_mu, val_vel, val_area)
        """
        grid = self.grid
        logteff = self.grid_teff
        logg = self.grid_logg
        wteff, jteff = self.Getaxispos(logteff,val_logteff)
        wlogg, jlogg = self.Getaxispos(logg,val_logg)
        # Here we convert val_vel to be the number of bins which
        # corresponds to the Doppler shift. That is z_obs / z_gridsampling.
        # This is more precise than just v_obs / v_gridsampling.
        val_vel = (numpy.sqrt((1+val_vel)/(1-val_vel)) - 1) / self.z
        #val_vel /= self.v
        jlam = numpy.floor(val_vel).astype('i')
        wlam = val_vel - jlam
        flux = Utils.Inter8_doppler_nomu(grid, wteff, wlogg, wlam, jteff, jlogg, jlam, val_area, val_mu*self.Limb_darkening(val_mu))
        return flux
    
    def __Make_log(self):
        """__Make_log()
        Resample the wavelength to be linear in log wavelength and
        recalculate the grid accordingly.
        
        >>> __Make_log()
        """
        new_grid_lam, self.v, self.z = Utils.Resample_linlog(self.grid_lam)
        ws, inds = Utils.Getaxispos_vector(self.grid_lam, new_grid_lam)
        self.grid_lam = new_grid_lam
        self.grid = self.grid.take(inds, axis=-1)*(1-ws) + self.grid.take(inds+1, axis=-1)*ws
        self.Coeff_limb_darkening(self.grid_lam/1e4)
        return
######################## class Atmo_grid_lithium_doppler ########################

