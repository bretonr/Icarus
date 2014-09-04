=======
Icarus
=======

---------------------------
Release history
---------------------------

## 2.0 (2014-08-11)

Features:

    - Modified atmosphere grid classes to a better management system derived from the astropy TableColumn class, itself a subclass of ndarray. Should allow for more object-oriented, pythonic approach.
    - Introduction of AtmoGrid, AtmoGridPhot and AtmoGridDoppler. These classes mainly utilise atmosphere grids saved into HDF5 format. They are self-contained with a proper header and all necessary meta data, and hence there is no need to provided a series of parameters when creating an instance.
    - Dropped support for pgplot. Now only support Matplotlib.

Bugfixes:

    - I have not tracked them all... but several small tweaks and improvements.

## 1.0 (2014-08-11)

Features:

    - First release of Icarus with the core functionalities.
