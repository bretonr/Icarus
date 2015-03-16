# Licensed under a 3-clause BSD style license - see LICENSE

import Icarus
from Icarus.Utils.import_modules import *


##### Welcome message
print( "##### example2.py Re-creating the Sun #####" )


##### Generating the Icarus object
ndiv = 5

print( "Creating an Icarus object" )
star = Icarus.Core.Star(ndiv, read=True)


##### Initializing the Icarus object with Vega's parameters
q = cts.Mearth / cts.Msun 
omega = 365.25/27 # This is roughly the ratio of the Sun's orbital period over its spin period
filling = 0.03
temp = 5778.
tempgrav = 0.25
tirr = 0.
porb = cts.SECPERJULYR # We use Earth as the binary companion so the orbital period is one year
k1 = cts.TWOPI * cts.astronomical_unit / porb * q # The Sun orbital velocity is the Earth's multiplied by the mass ratio
incl = 90.*cts.DEGTORAD

star.Make_surface(q=q, omega=omega, filling=filling, temp=temp, tempgrav=tempgrav, tirr=tirr, porb=porb, k1=k1, incl=incl)


##### Solving to get the filling factor which makes the solar radius unity
def solve_radius(f):
    star.Make_surface(filling=f)
    res = 1 - star.Radius() * star.separation / cts.Rsun
    return res

filling = scipy.optimize.bisect(solve_radius, 0.1*filling, 10*filling)


##### Loading various photometric bands
# Bessell photometric system
atmo_grid_U = Icarus.Atmosphere.Atmo_grid_BTSettl7('photometric_bands/BT-Settl.7.bessell.AB.U', 0.3578e-4, 0.0660e-4, 1.7898e-20, 5.530)
atmo_grid_B = Icarus.Atmosphere.Atmo_grid_BTSettl7('photometric_bands/BT-Settl.7.bessell.AB.B', 0.4355e-4, 0.0940e-4, 4.0626e-20, 4.700)
atmo_grid_V = Icarus.Atmosphere.Atmo_grid_BTSettl7('photometric_bands/BT-Settl.7.bessell.AB.V', 0.5466e-4, 0.0880e-4, 3.6375e-20, 3.550)
atmo_grid_R = Icarus.Atmosphere.Atmo_grid_BTSettl7('photometric_bands/BT-Settl.7.bessell.AB.R', 0.6473e-4, 0.1380e-4, 3.0648e-20, 2.660)
atmo_grid_I = Icarus.Atmosphere.Atmo_grid_BTSettl7('photometric_bands/BT-Settl.7.bessell.AB.I', 0.8007e-4, 0.1490e-4, 2.4166e-20, 1.700)
atmo_grid_J = Icarus.Atmosphere.Atmo_grid_BTSettl7('photometric_bands/BT-Settl.7.bessell.AB.J', 1.2275e-4, 0.2130e-4, 1.5893e-20, 1.000)
atmo_grid_H = Icarus.Atmosphere.Atmo_grid_BTSettl7('photometric_bands/BT-Settl.7.bessell.AB.H', 1.6365e-4, 0.3070e-4, 1.0214e-20, 0.624)
atmo_grid_K = Icarus.Atmosphere.Atmo_grid_BTSettl7('photometric_bands/BT-Settl.7.bessell.AB.K', 2.1979e-4, 0.3900e-4, 6.4032e-21, 0.382)
# SDSS photometric system
atmo_grid_u = Icarus.Atmosphere.Atmo_grid_BTSettl7('photometric_bands/BT-Settl.7.AB.u', 0.3541e-4, 0.0599e-4, 3.630e-20, 5.721)
atmo_grid_g = Icarus.Atmosphere.Atmo_grid_BTSettl7('photometric_bands/BT-Settl.7.AB.g', 0.4653e-4, 0.1379e-4, 3.630e-20, 4.207)
atmo_grid_r = Icarus.Atmosphere.Atmo_grid_BTSettl7('photometric_bands/BT-Settl.7.AB.r', 0.6147e-4, 0.1382e-4, 3.630e-20, 3.054)
atmo_grid_i = Icarus.Atmosphere.Atmo_grid_BTSettl7('photometric_bands/BT-Settl.7.AB.i', 0.7461e-4, 0.1535e-4, 3.630e-20, 2.315)
atmo_grid_z = Icarus.Atmosphere.Atmo_grid_BTSettl7('photometric_bands/BT-Settl.7.AB.z', 0.8904e-4, 0.1370e-4, 3.630e-20, 1.641)


##### Calculating the magnitude in various bands for the Sun
### Some useful litterature about the Sun's magnitude
## Christopher Willmer's: http://mips.as.arizona.edu/~cnaw/sun.html
## http://www.astro.umd.edu/~ssm/ASTR620/mags.html
## SDSS DR10: http://www.sdss3.org/dr10/algorithms/ugrizVegaSun.php
## http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
U = 6.36
B = 5.36
V = 4.82
R = 4.65
I = 4.55
J = 4.57
H = 4.71
K = 5.19
u = 6.45
g = 5.14
r = 4.65
i = 4.54
z = 4.52

## Determining the conversion factor between the apparent and absolute magnitude of the Sun
parsec = 3.085678e18 # parsec in cm
apparent_to_absolute = -2.5 * np.log10(((cts.astronomical_unit*100)/10./parsec)**2)

def Calc_mag(atmo_grid):
    mag = -2.5 * np.log10(star.Flux(0.0, atmo_grid=atmo_grid)/atmo_grid.flux0)
    return mag

sun_U = Calc_mag(atmo_grid_U) + apparent_to_absolute
sun_B = Calc_mag(atmo_grid_B) + apparent_to_absolute
sun_V = Calc_mag(atmo_grid_V) + apparent_to_absolute
sun_R = Calc_mag(atmo_grid_R) + apparent_to_absolute
sun_I = Calc_mag(atmo_grid_I) + apparent_to_absolute
sun_J = Calc_mag(atmo_grid_J) + apparent_to_absolute
sun_H = Calc_mag(atmo_grid_H) + apparent_to_absolute
sun_K = Calc_mag(atmo_grid_K) + apparent_to_absolute
sun_u = Calc_mag(atmo_grid_u) + apparent_to_absolute
sun_g = Calc_mag(atmo_grid_g) + apparent_to_absolute
sun_r = Calc_mag(atmo_grid_r) + apparent_to_absolute
sun_i = Calc_mag(atmo_grid_i) + apparent_to_absolute
sun_z = Calc_mag(atmo_grid_z) + apparent_to_absolute

print("diff. U: {:.3}".format(sun_U-U))
print("diff. B: {:.3}".format(sun_B-B))
print("diff. V: {:.3}".format(sun_V-V))
print("diff. R: {:.3}".format(sun_R-R))
print("diff. I: {:.3}".format(sun_I-I))
print("diff. J: {:.3}".format(sun_J-J))
print("diff. H: {:.3}".format(sun_H-H))
print("diff. K: {:.3}".format(sun_K-K))
print("diff. u: {:.3}".format(sun_u-u))
print("diff. g: {:.3}".format(sun_g-g))
print("diff. r: {:.3}".format(sun_r-r))
print("diff. i: {:.3}".format(sun_i-i))
print("diff. z: {:.3}".format(sun_z-z))


