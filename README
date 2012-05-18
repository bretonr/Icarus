=======
Icarus
=======

Icarus is a stellar binary light curve synthesis tool initially developed by Rene Breton while being a postdoctoral fellow at the University of Toronto in the research group of Marten van Kerkwijk.

Icarus provides a set of basic tools that:
1. generates a star given some basic binary parameters
 1.1. solves the gravitational potential equation
 1.2. creates a discretized stellar grid
 1.3. populates the stellar grid with physical parameters (temperature, surface gravity, etc.)
2. evaluates the outcoming flux from the star given an observer's point of view (i.e. orbtial phase and orbital orientation)

The code is compartimented in different layers:
1. the stellar surface solver
 1.1. the primitives generator of the discretized stellar grid: the code currently uses a triangular tessellation based the subdivision of an icosahedron. The primitives (vertices, face association, etc.) can be read from pre-calculated values stored in a file or dynamically generated using the external program "pygts" (http://pygts.sourceforge.net/), which is distributed separately and not essential to Icarus.
 1.2. the actual surface solver.
2. the flux calculator
 2.1. the actual flux calculator tools: integrated surface flux. Supported for different modes is already provided (spectroscopy, photometry, Doppler shifting, Doppler boosting).
 2.2. the flux calculator makes use of an atmosphere backend, which returns the specific intensities given a set of input parameters (temperature, surface gravity, velocity, etc.). The atmosphere backend can be anything (analytical blackbody, lookup table to an atmosphere model, etc.). The current backend reads data from NextGen atmosphere models (distributed separately).
3. the binary system super-class
 3.1. a super-class making use of the two above layers to treat with a "proper" binary, which sums the flux of each component and includes the calculation of eclipses, transits, partial occultations, etc.

The original aim of Icarus was to model the light curves (photometry and spectroscopy) of irradiated neutron star companions, hence the name Icarus (the Greek mythology hero who flew to close to the Sun and melting the wax off his wings). The flux calculator therefore supports the contribution of an external source of energy (from the other binary component) which contributes to increasing the dayside temperature of the modeled star.

Here are a short, non-exhaustive list of publications related to the binary light curve synthesis.
Breton et al., 2012, ApJL
Orosz, J. A., & Hauschildt, P. H. 2000, A&A, 364, 265
Hendry, P. D., & Mochnacki, S. W. 1992, ApJ, 388, 603


IF YOU USE IT:
If you intend to use the code, please cite the paper in which it was first introduced: R. P. Breton, S. A. Rappaport, M. H. van Kerkwijk, J. A. Carter. "KOI 1224, a Fourth Bloated Hot White Dwarf Companion Found With Kepler". ApJL, ???:??? April 2012.
Also, please provide a link to the Icarus webpage <https://github.com/bretonr/Icarus>.
The author (<superluminique + at + gmail.com>) would be happy to receive feedback, constructive comments, bug fixes, etc., from people using Icarus. Unfortunately, only very limited support can be provided due to the author's busy research schedule.


ACKNOWLEDGEMENTS:
Note that the author would like to acknowledge the immense help of Marten van Kerkwijk, who contributed via frequent discussions and who also provided a Fortran program to synthesize photometric light curves of irradiated binaries, which Icarus initially aimed to reproduce.


LICENSE:
Please note that this project is protected against a 3-clause BSD license. Please see the content of the folder "licenses" for more information.
