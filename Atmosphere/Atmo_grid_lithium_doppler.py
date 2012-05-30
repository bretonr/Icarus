# Licensed under a 3-clause BSD style license - see LICENSE

__all__ = ["Atmo_grid_lithium_doppler"]

from ..Utils.import_modules import *
from .. import Utils
from .Atmo_grid_lithium import Atmo_grid_lithium


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

