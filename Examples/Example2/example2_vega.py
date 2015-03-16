# Licensed under a 3-clause BSD style license - see LICENSE

import Icarus
from Icarus.Utils.import_modules import *


##### Welcome message
print( "##### example2_vega.py Re-creating Vega #####" )


##### Generating the Icarus object
ndiv = 5

print( "Creating an Icarus object" )
star = Icarus.Core.Star(ndiv, read=True)


##### Initializing the Icarus object with Vega's parameters, given a fake binary system
q = 0.01
omega = 1.0
filling = 0.7
temp = 10000.
tempgrav = 0.25
tirr = 0.
porb = 0.5*cts.SECPERDAY
k1 = 30.
incl = 5.7*cts.DEGTORAD

star.Make_surface(q=q, omega=omega, filling=filling, temp=temp, tempgrav=tempgrav, tirr=tirr, porb=porb, k1=k1, incl=incl)


##### ------------------------------------------------------------

##### Vega's parameters
Mass = 2.4
Rpole = 2.4
Req = 2.75
incl = 5.7*cts.DEGTORAD
Tpole = 10000.
Teq = Tpole - 1410.
logg_pole = 4.04
logg_eq = logg_pole - 0.26

##### Step 1: Find the filling factor given q, omega that makes r_eq/r_pole = 2.75/2.4
def solve_radius_ratio(filling, q=0.01, omega=1.0):
    star.Make_surface(q=q, omega=omega, filling=filling)
    return Req/Rpole - 0.5*(star.rc_l1 + star.rc_eq)/star.rc_pole

q = 0.001
omega = 1.0
filling = scipy.optimize.brentq(solve_radius_ratio, 0.1, 1.0, args=(q,omega))

##### Step 2: We solve for the orbital period given the rc_eq found from step 1 that makes the mass 2.4Msun and the equatorial radius 2.4Rsun
porb = np.sqrt( (4*cts.PI**2) / (cts.G * Mass*cts.Msun * (1+q)) * (Req*cts.Rsun/star.rc_eq)**3 )

##### Step 3: We solve for the projected velocity given the orbital period found from step 2 that makes everything else self-consistent
K = ( Mass*cts.Msun * 2*cts.PI * cts.G * np.sin(incl)**3 * q**3 / (1+q)**2 / porb )**(1./3)

##### Updating the star object
star.Make_surface(q=q, omega=omega, filling=filling, temp=Tpole, porb=porb, k1=K, incl=incl)

##### Step 4: We solve for the gravity darkening coefficient that yields the right temperatures
def solve_gravity(tempgrav):
    star.Make_surface(tempgrav=tempgrav)
    return (Tpole-Teq) - (np.exp(star.logteff.max()) - np.exp(star.logteff.min()))

tempgrav = scipy.optimize.brentq(solve_gravity, 0.2, 0.3)

##### Updating the star object
star.Make_surface(q=q, omega=omega, filling=filling, temp=Tpole, tempgrav=tempgrav, porb=porb, k1=K, incl=incl)

##### Printing the input parameters
print( "" )
print( "{:^60}".format('Table of input parameters for Vega') )
print( "{:40} {:>10.4f}".format('q',q) )
print( "{:40} {:>10.4f}".format('omega',omega) )
print( "{:40} {:>10.4f}".format('filling',filling) )
print( "{:40} {:>10.1f}".format('Tpole',Tpole) )
print( "{:40} {:>10.3f}".format('tempgrav',tempgrav) )
print( "{:40} {:>10.2f}".format('porb',porb) )
print( "{:40} {:>10.4f}".format('K',K) )
print( "{:40} {:>10.4f}".format('incl',incl) )
print( "{:40} {:>10.1f}".format('tirr',tirr) )


##### Printing some basic information
print( "" )
print( "{:^60}".format('Table of parameters for Vega') )
print( "{:40} {:>10} {:>10}".format('','Paper','Model') )
print( "{:40} {:>10.2f} {:>10.2f}".format('Rpole (Rsun)',Rpole,star.rc_pole*star.separation/cts.Rsun) )
print( "{:40} {:>10.2f} {:>10.2f}".format('Req (Rsun)',Req,0.5*(star.rc_eq+star.rc_l1)*star.separation/cts.Rsun) )
print( "{:40} {:>10.0f} {:>10.0f}".format('Tpole (K)',Tpole,np.exp(star.logteff.max())) )
print( "{:40} {:>10.0f} {:>10.0f}".format('Teq (K)',Teq,np.exp(star.logteff.min())) )
print( "{:40} {:>10.2f} {:>10.2f}".format('Mass (Msun)',Mass,star.mass1) )
print( "{:40} {:>10.3f} {:>10.3f}".format('Pole-to-equator log(g) (cgx, dex)',logg_pole-logg_eq,star.logg_pole-star.logg_eq) )


##### ------------------------------------------------------------


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
Vvega = 0.03
#V = Vvega - 0.044
#U = Vvega - 0.163
#B = Vvega - 0.139
#R = Vvega + 0.117
#I = Vvega + 0.342
V = Vvega
U = Vvega
B = Vvega
R = Vvega
I = Vvega
J = Vvega
H = Vvega
K = Vvega

g = -0.08
u = g + 1.02
r = g + 0.25
i = r + 0.23
z = i + 0.17

## Determining the conversion factor between the apparent and absolute magnitude of the Sun
distance = 7.678 # in parsec (from Hipparcos, see Linnell, DeStefano & Hubzny, 2013, ApJ, 146, 68)
apparent_to_absolute = -2.5 * np.log10((distance/10)**2)

def Calc_mag(atmo_grid):
    mag = star.Mag_flux(0.0, atmo_grid=atmo_grid)
    return mag

mag_U = Calc_mag(atmo_grid_U) - apparent_to_absolute
mag_B = Calc_mag(atmo_grid_B) - apparent_to_absolute
mag_V = Calc_mag(atmo_grid_V) - apparent_to_absolute
mag_R = Calc_mag(atmo_grid_R) - apparent_to_absolute
mag_I = Calc_mag(atmo_grid_I) - apparent_to_absolute
mag_J = Calc_mag(atmo_grid_J) - apparent_to_absolute
mag_H = Calc_mag(atmo_grid_H) - apparent_to_absolute
mag_K = Calc_mag(atmo_grid_K) - apparent_to_absolute
mag_u = Calc_mag(atmo_grid_u) - apparent_to_absolute
mag_g = Calc_mag(atmo_grid_g) - apparent_to_absolute
mag_r = Calc_mag(atmo_grid_r) - apparent_to_absolute
mag_i = Calc_mag(atmo_grid_i) - apparent_to_absolute
mag_z = Calc_mag(atmo_grid_z) - apparent_to_absolute

print("")
print("diff. U: {:.3}".format(mag_U-U))
print("diff. B: {:.3}".format(mag_B-B))
print("diff. V: {:.3}".format(mag_V-V))
print("diff. R: {:.3}".format(mag_R-R))
print("diff. I: {:.3}".format(mag_I-I))
print("diff. J: {:.3}".format(mag_J-J))
print("diff. H: {:.3}".format(mag_H-H))
print("diff. K: {:.3}".format(mag_K-K))
print("diff. u: {:.3}".format(mag_u-u))
print("diff. g: {:.3}".format(mag_g-g))
print("diff. r: {:.3}".format(mag_r-r))
print("diff. i: {:.3}".format(mag_i-i))
print("diff. z: {:.3}".format(mag_z-z))


