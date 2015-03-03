# Icarus

## Release history

### 2.1.x (2015-02-20)

Features:

Bug fixes:

- Fixed an issue with the conversion of mag to/from flux. The new atmosphere grids should have zeropoints, rather than _flux0_.


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


