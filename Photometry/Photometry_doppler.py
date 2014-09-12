# Licensed under a 3-clause BSD style license - see LICENSE

__all__ = ["Photometry_doppler"]

from ..Utils.import_modules import *
from .. import Utils
from .. import Core
from .. import Atmosphere
from .Photometry import Photometry


######################## class Photometry ########################
class Photometry_doppler(Photometry):
    """Photometry_doppler(Photometry)
    This class allows to fit the flux from the primary star
    of a binary system, assuming it is heated by the secondary
    (which in most cases will be a pulsar).
    
    It is meant to deal with photometry data. Many sets of photometry
    data (i.e. different filters) are read. For each data set, one can
    calculate the predicted flux of the model at every data point (i.e.
    for a given orbital phase).

    This subclass of Photometry includes Doppler boosting calculation.
    """
    def __init__(self, atmo_fln, data_fln, ndiv, porb, x2sini, edot=1., read=True):
        """__init__(atmo_fln, data_fln, ndiv, porb, x2sini, edot=1., read=True)
        This class allows to fit the flux from the primary star
        of a binary system, assuming it is heated by the secondary
        (which in most cases will be a pulsar).
        
        It is meant to deal with photometry data. Many sets of photometry
        data (i.e. different filters) are read. For each data set, one can
        calculate the predicted flux of the model at every data point (i.e.
        for a given orbital phase).
        
        atmo_fln (str): A file containing the grid model information for each
            data set. The format of each line of the file is as follows:
                band_name, center_wavelength, delta_wavelength, flux0,
                    extinction, grid_file
        data_fln (str): A file containing the information for each data set.
            The format of the file is as follows:
                band_name, column_phase, column_flux, column_error_flux,
                    shift_to_phase_zero, calibration_error, softening_asinh,
                    data_file
            Here, the first column has index 0.
            Here, orbital phase 0. is the superior conjunction of the pulsar.
        ndiv (int): The number of surface slice. Defines how coarse/fine the
            surface grid is.
        porb (float): Orbital period of the system in seconds.
        x2sini (float): Projected semi-major axis of the secondary (pulsar)
            in light-second.
        edot (float): Irradiated energy from the secondary, aka pulsar (i.e.
            spin-down luminosity) in erg/s. This is only used for the
            calculation of the irradiation efficiency so it does not
            enter in the modeling itself.
        read (bool): If True, Icarus will use the pre-calculated geodesic
            primitives. This is the recommended option, unless you have the
            pygts package installed to calculate it on the spot.
        
        >>> fit = Photometry(atmo_fln, data_fln, ndiv, porb, x2sini)
        """
        super(Photometry_doppler, self).__init__(atmo_fln, data_fln, ndiv, porb, x2sini, edot=edot, read=read)

    def Get_flux(self, par, flat=False, func_par=None, DM_AJ=False, nsamples=None, verbose=False):
        """Get_flux(par, flat=False, func_par=None, DM_AJ=False, nsamples=None, verbose=False)
        Returns the predicted flux (in magnitude) by the model evaluated
        at the observed values in the data set.
        
        par: Parameter list.
            [0]: Orbital inclination in radians.
            [1]: Corotation factor.
            [2]: Roche-lobe filling.
            [3]: Companion temperature.
            [4]: Gravity darkening coefficient.
            [5]: K (projected velocity semi-amplitude) in m/s.
            [6]: Front side temperature or irradiation temperature.
                The irradiation temperature is in the case of the
                photometry_modeling_temperature class.
            [7]: Distance modulus (optional).
            [8]: Absorption A_J (optional).
            Note: Can also be a dictionary:
                par.keys() = ['aj', 'corotation', 'dm', 'filling',
                    'gravdark', 'incl','k1','tday','tnight']
        flat (False): If True, the values are returned in a 1D vector.
            If False, predicted values are grouped by data set left in a list.
        func_par (None): Function that takes the parameter vector and
            returns the parameter vector. This allow for possible constraints
            on the parameters. The vector returned by func_par must have a length
            equal to the number of expected parameters.
        DM_AJ (False): If true, will include the DM and AJ in the flux.
        nsamples (None): Number of points for the lightcurve sampling.
            If None, the lightcurve will be sampled at the observed data
            points.
        
        Note: tirr = (par[6]**4 - par[3]**4)**0.25
        
        >>> self.Get_flux([PIBYTWO,1.,0.9,4000.,0.08,300e3,5000.,10.,0.])
        """
        # func_par
        if func_par is not None:
            par = func_par(par)
        # check if we are dealing with a dictionary
        if isinstance(par, dict):
            par = [par['incl'], par['corotation'], par['filling'], par['tnight'], par['gravdark'], par['k1'], par['tday'], par['dm'], par['aj']]
        
        # We call Make_surface to make the companion's surface.
        self.Make_surface(par, verbose=verbose)
        
        # If nsamples is None we evaluate the lightcurve at each data point.
        if nsamples is None:
            phases = self.data['phase']
        # If nsamples is set, we evaluate the lightcurve at nsamples
        else:
            phases = (numpy.arange(nsamples, dtype=float)/nsamples).repeat(self.ndataset).reshape((nsamples,self.ndataset)).T
        
        # If DM_AJ, we take into account the DM and AJ into the flux here.
        if DM_AJ:
            DM_AJ = self.data['ext']*par[8] + par[7]
        else:
            DM_AJ = self.data['ext']*0.
        
        # Calculate the actual lightcurves
        flux = []
        for i in numpy.arange(self.ndataset):
                # If we use the interpolation method and if the filter is the same as a previously
                # calculated one, we do not recalculate the fluxes and simply copy them.
                if nsamples is not None and self.grouping[i] < i:
                    flux.append(flux[self.grouping[i]])
                else:
                    flux.append( numpy.array([self.star.Mag_flux_doppler(phase, atmo_grid=self.atmo_grid[i], atmo_doppler=self.atmo_doppler[i]) for phase in phases[i]]) + DM_AJ[i] )
        
        # If nsamples is set, we interpolate the lightcurve at nsamples.
        if nsamples is not None:
            for i in numpy.arange(self.ndataset):
                ws, inds = Utils.Series.Getaxispos_vector(phases[i], self.data['phase'][i])
                flux[i] = flux[i][inds]*(1-ws) + flux[i][inds+1]*ws
        
        # We can flatten the flux array to simplify some of the calculations in the Calc_chi2 function
        if flat:
            return numpy.hstack(flux)
        else:
            return flux

    def Get_flux_theoretical(self, par, phases, func_par=None, verbose=False):
        """Get_flux_theoretical(par, phases, func_par=None, verbose=False)
        Returns the predicted flux (in magnitude) by the model evaluated at the
        observed values in the data set.
        
        par: Parameter list.
            [0]: Orbital inclination in radians.
            [1]: Corotation factor.
            [2]: Roche-lobe filling.
            [3]: Companion temperature.
            [4]: Gravity darkening coefficient.
            [5]: K (projected velocity semi-amplitude) in m/s.
            [6]: Front side temperature or irradiation temperature.
                The irradiation temperature is in the case of the
                photometry_modeling_temperature class.
            [7]: Distance modulus.
            [8]: Absorption A_J.
            Note: Can also be a dictionary:
                par.keys() = ['aj','corotation','dm','filling','gravdark','incl','k1','tday','tnight']
        phases: A list of orbital phases at which the model should be
            evaluated. The list must have the same length as the
            number of data sets, each element can contain many phases.
        func_par (None): Function that takes the parameter vector and
            returns the parameter vector. This allow for possible constraints
            on the parameters. The vector returned by func_par must have a length
            equal to the number of expected parameters.
        verbose (False)
        
        Note: tirr = (par[6]**4 - par[3]**4)**0.25
        
        >>> self.Get_flux_theoretical([PIBYTWO,1.,0.9,4000.,0.08,300e3,5000.,10.,0.], [[0.,0.25,0.5,0.75]]*4)
        """
        # func_par
        if func_par is not None:
            par = func_par(par)
        # check if we are dealing with a dictionary
        if isinstance(par, dict):
            par = [par['incl'], par['corotation'], par['filling'], par['tnight'], par['gravdark'], par['k1'], par['tday'], par['dm'], par['aj']]
        
        # We call Make_surface to make the companion's surface.
        self.Make_surface(par, verbose=verbose)
        
        DM_AJ = self.data['ext']*par[8] + par[7]
        
        flux = []
        for i in numpy.arange(self.ndataset):
            # If the filter is the same as a previously calculated one
            # we do not recalculate the fluxes and simply copy them.
            if self.grouping[i] < i:
                flux.append( flux[self.grouping[i]] )
            else:
                flux.append( numpy.array([self.star.Mag_flux_doppler(phase, atmo_grid=self.atmo_grid[i], atmo_doppler=self.atmo_doppler[i]) for phase in phases[i]]) + DM_AJ[i] )
        return flux

    def Get_Keff(self, *args, **kwargs):
        """
        Returns the effective projected velocity semi-amplitude of the star in m/s.
        The luminosity-weighted average velocity of the star is returned for
        nphases, for the specified dataset, and a sin wave is fitted to them.
        
        par: Parameter list.
            [0]: Orbital inclination in radians.
            [1]: Corotation factor.
            [2]: Roche-lobe filling.
            [3]: Companion temperature.
            [4]: Gravity darkening coefficient.
            [5]: K (projected velocity semi-amplitude) in m/s.
            [6]: Front side temperature.
            [7]: Distance modulus.
            [8]: Absorption A_J.
        nphases (int): Number of phases to evaluate the velocity at.
        atmo_grid (int, AtmoGridPhot): The atmosphere grid to use for the velocity
            calculation. Can be an integer that represents the index of the atmosphere
            grid object in self.atmo_grid, and it can be an AtmoGridPhot instance.
        func_par (function): Function that takes the parameter vector and
            returns the parameter vector. This allow for possible constraints
            on the parameters. The vector returned by func_par must have a length
            equal to the number of expected parameters.
        make_surface (bool): Whether lightcurve.make_surface should be called
            or not. If the flux has been evaluate before and the parameters have
            not changed, False is fine.
        verbose (bool): Verbosity. Will plot the velocities and the sin fit.
        """
        warnings.warn('Careful. No Doppler boosting is applied for the calculation of Get_Keff.')
        return super(Photometry_doppler, self).Get_Keff(*args, **kwargs)

    def _Read_atmo(self, atmo_fln):
        """_Read_atmo(atmo_fln)
        Reads the atmosphere model data.
        
        atmo_fln: A file containing the grid model information for each
            data set. The format of each line of the file is as follows:
                band_name, grid_file, doppler_file
        
        >>> self._Read_atmo(atmo_fln)
        """
        f = open(atmo_fln,'r')
        lines = f.readlines()
        self.atmo_grid = []
        self.atmo_doppler = []
        for line in lines:
            if (line[0] != '#') and (line[0] != '\n'):
                tmp = line.split()
                self.atmo_grid.append(Atmosphere.AtmoGridPhot.ReadHDF5(tmp[1]))
                self.atmo_grid.append(Atmosphere.AtmoGridDoppler.ReadHDF5(tmp[2]))
        return

######################## class Photometry_doppler ########################

