# Licensed under a 3-clause BSD style license - see LICENSE

__all__ = ["Star_disk"]

from ..Utils.import_modules import *
from .. import Utils
from .Star import Star


######################## class Star_disk ########################
class Star_disk(Star):
    """ Star_disk(Star)
    This class allows to determine the flux of a star
    in a binary system using an atmosphere grid. It is derived
    from the Star class.
    
    The noticeable difference is that this class contains the tools
    to add the flux contribution from a disk in the system.
    For the moment the contribution is restricted to a constant flux.
    """
    def __init__(self, *args, **kwargs):
        Star.__init__(self, *args, **kwargs)

    def Flux_disk(self, phase, gravscale=None, atmo_grid=None, disk=0.):
        """Flux(phase, gravscale=None, atmo_grid=None, disk=0.)
        Return the flux interpolated from the atmosphere grid.
        Adds a constant flux contribution due to a disk.
        
        phase: orbital phase (in orbital fraction; 0: companion 
            in front, 0.5: companion behind).
        gravscale (optional): gravitational scaling parameter.
        atmo_grid (optional): atmosphere grid instance used to
            calculate the flux.
        
        >>> self.Flux(phase)
        flux
        """
#        print( "Flux_disk" )
        if atmo_grid is None:
            atmo_grid = self.atmo_grid
        if gravscale is None:
            gravscale = self._Gravscale()
        mu = self._Mu(phase)
        inds = (mu > 0).nonzero()[0]
#        fsum = 0.
#        for i in inds:
#            fsum = fsum + self.area[i] * atmo_grid.Get_flux(self.logteff[i],self.logg[i]+gravscale,mu[i])
#        if fsum.ndim == 2:
#            fsum = self.area[inds].reshape((inds.size,1)) * fsum
#        else:
#            fsum = self.area[inds] * fsum
#        fsum = fsum.sum(axis=0)
        fsum = atmo_grid.Get_flux(self.logteff[inds],self.logg[inds]+gravscale,mu[inds],self.area[inds])
        return fsum + disk

    def Flux_disk_Keff(self, phase, gravscale=None, atmo_grid=None, disk=0.):
        """Flux_disk_Keff(phase, gravscale=None, atmo_grid=None, disk=0.)
        Return the flux interpolated from the atmosphere grid.
        Adds a constant flux contribution due to a disk.
        Also returns the effective velocity of the star.
        
        phase: orbital phase (in orbital fraction; 0: companion 
            in front, 0.5: companion behind).
        gravscale (optional): gravitational scaling parameter.
        atmo_grid (optional): atmosphere grid instance used to
            calculate the flux.
        
        >>> self.Flux_disk_Keff(phase)
        flux, Keff
        """
#        print( "Flux_disk_Keff" )
        if atmo_grid is None:
            atmo_grid = self.atmo_grid
        if gravscale is None:
            gravscale = self._Gravscale()
        mu = self._Mu(phase)
        v = self._Velocity_surface(phase)
        inds = (mu > 0).nonzero()[0]
#        fsum = 0.
#        for i in inds:
#            fsum = fsum + self.area[i] * atmo_grid.Get_flux(self.logteff[i],self.logg[i]+gravscale,mu[i])
#        if fsum.ndim == 2:
#            fsum = self.area[inds].reshape((inds.size,1)) * fsum
#        else:
#            fsum = self.area[inds] * fsum
#        fsum = fsum.sum(axis=0)
        fsum, Keff = atmo_grid.Get_flux_Keff(self.logteff[inds],self.logg[inds]+gravscale,mu[inds],self.area[inds],v[inds])
        return fsum + disk, Keff*cts.c

    def Mag_flux_disk(self, phase, gravscale=None, a=None, atmo_grid=None, disk=0.):
        """Mag_flux_disk(phase, gravscale=None, a=None, disk=0.)
        Returns the magnitude interpolated from the atmosphere grid.
        The flux is added to a constant quantity, which simulates the contribution
        from an overly simplistic accretion disk.
        
        phase: orbital phase (in orbital fraction; 0: companion 
            in front, 0.5: companion behind).
        gravscale (optional): gravitational scaling parameter.
        a (optional): orbital separation. If not provided, derives it
            from q and asini (provided when creating the instance of
            the class).
        atmo_grid (optional): atmosphere grid instance to work from to 
            calculate the flux.
        disk (=0.): the contribution from the accretion disk around the neutron star.
            It is assumed to be a constant flux value that is added to the companion's
            emission.
        
        >>> self.Mag_flux_disk(phase)
        mag_flux_disk
        """
#        print( "Mag_flux_disk" )
        if atmo_grid is None:
            atmo_grid = self.atmo_grid
        if a is not None:
            proj = self._Proj(a)
        else:
            proj = self._Proj(self.separation)
        if gravscale is None:
            gravscale = self._Gravscale()
        return -2.5*np.log10(self.Flux_disk(phase, gravscale=gravscale, atmo_grid=atmo_grid, disk=disk) * proj) + atmo_grid.meta['zp']

######################## class Star_disk ########################

