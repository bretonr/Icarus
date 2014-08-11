# Licensed under a 3-clause BSD style license - see LICENSE

import Icarus
from Icarus.Utils.import_modules import *


##### This creates a star and calculates the flux at a given orbital phase.
##### The result from a calculation done step by step is compared to the Flux function.


atmo = Icarus.Atmosphere.AtmoGridPhot.ReadHDF5('V.h5')
star = Icarus.Core.Star_temperature(5)

q = 90.1
omega = 1.0
filling = 0.05
temp = 4500.
tempgrav = 0.25
tirr = 4500.
porb = cts.SECPERDAY
k1 = 300e3
incl = cts.pibytwo

star.Make_surface(q=q, omega=omega, filling=filling, temp=temp, tempgrav=tempgrav, tirr=tirr, porb=porb, k1=k1, incl=incl)
flux_ref = star.Flux(0., atmo_grid=atmo)
print('Flux (function): {}'.format(flux_ref))

gravscale = star._Gravscale()
mu = star._Mu(0.)
inds = mu > 0
logteff = star.logteff[inds]
logg = star.logg[inds]+gravscale
mu = mu[inds]
area = star.area[inds]
flux1 = atmo.Get_flux(logteff, logg, mu, area)
print('Flux (step by step, 1): {}'.format(flux1))

w1temp, jtemp = atmo.Getaxispos('logtemp', logteff)
w1logg, jlogg = atmo.Getaxispos('logg', logg)
w1mu, jmu = atmo.Getaxispos('mu', mu)
flux2 = Icarus.Utils.Grid.Inter8_photometry(atmo.data, w1temp, w1logg, w1mu, jtemp, jlogg, jmu, area, mu)

print('Flux (step by step, 2): {}'.format(flux2))






