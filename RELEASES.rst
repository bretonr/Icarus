# Icarus

## Release history

### 2.3.0 (2016-04-15)

Features:

- This major release has now changed the directory structure to be in line with the standard Python packaging. It includes a ``setup.py`` to enable the installation of the package to the standard Python library. There is also a PyPI distribution now. Many thanks to **@muddassir235** for the pull request!


Bug fixes:

- _Photometry/Photometry.py/Photometry_temperature_: Updated the class so is now compatible with the new version of the _Photometry_ class.


### 2.2.2 (2015-10-12)

Bug fixes:

- _Atmosphere/Atmo.py/AtmoGridSpec.Get_flux_doppler_: There was an issue with the interpolation along the wavelength axis which has now been resolved. It was simply not computed correctly.
- _Core/Star_base_: Fixed a problem in the Velocity_surface function which calculates the surface velocity.
- _Atmosphere/Atmo_spectro_IRTF.py_: Fixed the conversion of the fluxes, which must be stored in the grid and log(F_lambda), but the grid fluxes store them at log10(F_lambda).
- _Utils/Grid.py_: _Interp_doppler_savememory_ and _Inter_doppler_ were both not exponentiating the fluxes before summing them over the surface elements. One needs to bear in mind that the grid values are log(F_lambda).


### 2.2.1 (2015-10-12)

Features:

- _Atmosphere/Atmo.py/AtmoGridSpec_ is a new class to handle atmosphere grids in the spectral domain.

- _Utils/Filter.py/Band_integration_ is modified in order to handle masking parts of the spectral range. The default method is Simpson's rule, but trapezoid is also implemented and recommended for the masking.

### 2.2.0 (2015-10-09)

Features:

- _Atmosphere/Atmo.py/AtmoGrid_

    - New _Fill_nan_ function to fill non-existing values.
    - New _SubGrid_ function to return a sub-grid of the atmosphere grid.
    - New _IsFinite_ function to return a boolean view of existing values

- _Utils/Binary.py_: New function to calculate the approximate Roche lobe radius using the Eggleton formula.
- _Utils/Filter.py/Load_filter_: New option that can also return the min/max bounds of the filter, in addition to the filter interpolator.


Bug fixes:

- _Core/Star.py/Make_surface_

    - Changed the way that coschi -- angle between the irradiation source and the normal to the surface -- is calculated. It used to be computed using the spherical approximation (i.e that the normal to the surface was roughly the same as the vector direction from the centre of mass of the star to the surface). This worked well for low filling factors, but breaks down at larger filling factor. It is better to use the exact value of the normal to the surface, which is the gradient vector (already calculated to solve the equipotential surface). Because the irradiation source is along the x-direction, coschi simply corresponds to the x-component of the gradient vector. __Many thanks to Roger Romani and Nicholas Sanchez at Stanford University for highlighting this issue and providing pieces of codes.__

- Small bug fix in _Photometry/Photometry.py_ regarding a typo in a variable name for the _full_output_ option in the _Plot_model_ function.


### 2.1.2 (2015-03-18)

Features:

- In the _Star_base_ class, change in the way _Flux_ is calculated in order to add the possibility to account for the projection effect in order to scale the flux to physical values. There wasn't a need for it before as the fluxes were always used in a normalised way (e.g. fitting a non-flux calibrated spectrum), but this would make things fail if one was to fit flux calibrated data. The issue wouldn't arise with magnitudes, which already included the projection.

Bug fixes:

- Fixed an issue with the conversion of mag to/from flux in the _Photometry_ module. The conversion was expecting to find a _flux0_ parameter, whereas the zeropoint is now incorporated as _zp_ in the atmosphere files.
- The _Utils.import_modules_ has a set of constants defined under the _cts_ namespace. Converted some lower case constants into upper case.
- Many changes made to the _Photometry_ module in order to handle working in the flux space properly (as opposed to the magnitude space). Now all the plotting can be done in the flux space. Also, everything related to dealing with the band offset has been rebrewed in order to simplify function calls and make everything more intuitive. This includes changes to the optional arguments and their names. Please make sure to look at the docstrings for _Calc_chi2_, _Get_flux_, _Get_flux_theoretical_, _Plot_.


### 2.1.1 (2015-02-20)

Features:

- New _Distance_to_distance_modulus_ and _Distance_modulus_to_distance_ functions in the _Utils.Flux_ module.
- The Photometry class in the Photometry module has been updated significantly.
    - The initialisation does not take porb, x2sini anymore.
    - Calling Calc_chi2, Get_flux, etc, uses a new list of parameters, which calls the function Photometry Make_surface directly. The point is to reflect more closely the Star base class and allow for modification of the parameters more easily (e.g. in case the orbital period is a free parameter).
    - The DM and AV are passed to the functions as DM and AV optional parameters rather than through the list of parameters.
    - IMPORTANT NOTE: In order to make your old scripts still work, change any call from Icarus.Photometry.Photometry to Icarus.Photometry_legacy.Photometry_legacy in order to use the old version.

Bug fixes:

- Changed the Atmo_IRTF_spectro class name to Atmo_spectro_IRTF to match the module name.


### 2.1.0 (2014-11-21)

Features:

- MAJOR: Changed the nomenclature of extinction to be A_V instead of A_J. V-band extinction is much more common than J-band.
- Created a new input format for the data file. The old 7-column and 8-column formats are still supported. However the new 9-column format includes a keyword that allows to switch the input between magnitude and flux.

Bug fixes:

- Fixed an issue with class inheritence needing to descend from "object" rather than being blank, in order to use the "super" parent function call.


### 2.0.2 (2014-10-10)

Bug fixes:

- Fixed an issue with the sign of the velocity (being negative) returned by Photometry.Get_Keff.


### 2.0 (2014-08-11)

Features:

- Modified atmosphere grid classes to a better management system derived from the astropy `TableColumn` class, itself a subclass of ndarray. Should allow for more object-oriented, pythonic approach.
- Introduction of `AtmoGrid`, `AtmoGridPhot` and `AtmoGridDoppler`. These classes mainly utilise atmosphere grids saved into HDF5 format. They are self-contained with a proper header and all necessary meta data, and hence there is no need to provided a series of parameters when creating an instance.
- Dropped support for pgplot. Now only support Matplotlib.

Bug fixes:

- I have not tracked them all... but several small tweaks and improvements.


### 1.0 (2014-08-11)

Features:

- First release of **Icarus** with the core functionalities.


