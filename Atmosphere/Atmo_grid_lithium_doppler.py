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
    def __init__(self, flns, oversample=None, smooth=None, thin=None, convert=None, zp=0., wave_cut=[3000,11000], verbose=False, savememory=True):
        """
        """
        if verbose: print( 'Reading atmosphere grid files' )
        #Atmo_grid_lithium.__init__(self, flns, oversample=oversample, smooth=smooth, thin=thin, convert=convert, zp=zp, wave_cut=wave_cut, linlog=True, verbose=verbose, savememory=savememory)
        Atmo_grid_lithium.__init__(self, flns, oversample=oversample, smooth=smooth, thin=thin, convert=convert, zp=zp, wave_cut=wave_cut, linlog=False, verbose=verbose, savememory=savememory)
        #print( 'Rebinning to linear in logarithmic spacing' )
        #self.__Make_log()
    
    def Get_flux_doppler(self, val_logtemp, val_logg, val_mu, val_area, val_vel, **kwargs):
        """
        Returns the flux interpolated from the atmosphere grid.
        val_logtemp: log of effective temperature
        val_logg: log of surface gravity
        val_mu: cos(angle) of angle of emission
        val_vel: velocity of the grid point in units of speed of light
        val_area: area of the surface element
        
        >>> flux = self.Get_flux_doppler(val_logtemp, val_logg, val_mu, val_area, val_vel)
        """
        grid = self.grid
        logtemp = self.logtemp
        logg = self.logg
        mu = self.mu
        wtemp, jtemp = self.Getaxispos(logtemp,val_logtemp)
        wlogg, jlogg = self.Getaxispos(logg,val_logg)
        wmu, jmu = self.Getaxispos(mu,val_mu)
        # Here we convert val_vel to be the number of bins which
        # corresponds to the Doppler shift. That is z_obs / z_gridsampling.
        # This is more precise than just v_obs / v_gridsampling.
        #val_vel = (np.sqrt((1+val_vel)/(1-val_vel)) - 1) / self.z
        #val_vel /= self.v
        #jwav = np.floor(val_vel).astype('i')
        #wwav = val_vel - jwav
        if self.savememory:
            grid_mu = self.grid_mu
            if self.z0:
                flux = Utils.Grid.Interp_doppler_savememory_linear(grid, wtemp, wlogg, wmu, jtemp, jlogg, jmu, grid_mu, val_area, val_mu, val_vel, self.z0)
            else:
                print( 'Hey! Wake up! The grid is not linear in lambda and has been transformed to linear in log(lambda)!' )
        else:
            flux = Utils.Grid.Interp_doppler(grid, wtemp, wlogg, wmu, wwav, jtemp, jlogg, jmu, jwav, val_area, val_mu)
        return flux
    
    def Get_flux_doppler_nomu(self, val_logtemp, val_logg, val_mu, val_area, val_vel, **kwargs):
        """
        Returns the flux interpolated from the atmosphere grid.
        val_logtemp: log of effective temperature
        val_logg: log of surface gravity
        val_mu: cos(angle) of angle of emission
        val_area: area of the surface element
        val_vel: velocity of the grid point in units of speed of light
        
        >>> flux = self.Get_flux_doppler_nomu(val_logtemp, val_logg, val_mu, val_area, val_vel)
        """
        grid = self.grid
        logtemp = self.logtemp
        logg = self.logg
        wtemp, jtemp = self.Getaxispos(logtemp,val_logtemp)
        wlogg, jlogg = self.Getaxispos(logg,val_logg)
        # Here we convert val_vel to be the number of bins which
        # corresponds to the Doppler shift. That is z_obs / z_gridsampling.
        # This is more precise than just v_obs / v_gridsampling.
        val_vel = (np.sqrt((1+val_vel)/(1-val_vel)) - 1) / self.z
        #val_vel /= self.v
        jwav = np.floor(val_vel).astype('i')
        wwav = val_vel - jwav
        flux = Utils.Grid.Interp_doppler_nomu(grid, wtemp, wlogg, wwav, jtemp, jlogg, jwav, val_area, val_mu*self.Limb_darkening(val_mu))
        return flux
    
    def __Make_log(self):
        """__Make_log()
        Resample the wavelength to be linear in log wavelength and
        recalculate the grid accordingly.
        
        >>> __Make_log()
        """
        new_wav, self.v, self.z = Utils.Series.Resample_linlog(self.wav)
        ws, inds = Utils.Series.Getaxispos_vector(self.wav, new_wav)
        self.wav = new_wav
        self.grid = self.grid.take(inds, axis=-1)*(1-ws) + self.grid.take(inds+1, axis=-1)*ws
        self.Coeff_limb_darkening(self.wav/1e4)
        return
######################## class Atmo_grid_lithium_doppler ########################

