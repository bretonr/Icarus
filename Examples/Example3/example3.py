# Licensed under a 3-clause BSD style license - see LICENSE

import Icarus
from Icarus.Utils.import_modules import *


##### This creates a star and calculates the flux at a given orbital phase.
##### The result from a calculation done step by step is compared to the Flux function.


atmo = Icarus.Atmosphere.AtmoGridPhot.ReadHDF5('photometric_bands/AGSS/AGSS.Bessell.V.h5')
atmo_doppler = Icarus.Atmosphere.AtmoGridDoppler.ReadHDF5('photometric_bands/AGSS/Doppler.AGSS.Bessell.V.h5')
star = Icarus.Core.Star_temperature(5)

phs = np.linspace(0., 1., 51)


## Testing Doppler boosting with an non-irradiated, undistorted star
q = 100.
omega = 1.0
filling = 0.01
temp = 4500.
tempgrav = 0.25
tirr = 0.
porb = cts.SECPERDAY
k1 = 500e3
incl = cts.PIBYTWO

star.Make_surface(q=q, omega=omega, filling=filling, temp=temp, tempgrav=tempgrav, tirr=tirr, porb=porb, k1=k1, incl=incl)

nosum = False
flux_regular = np.array([star.Flux(p, atmo_grid=atmo, nosum=nosum) for p in phs])
flux_doppler = np.array([star.Flux_doppler(p, atmo_grid=atmo, atmo_doppler=atmo_doppler, nosum=nosum) for p in phs])
print('{:-^80}'.format(' Comparing flux and flux_doppler' ))
print('Regular')
Pprint(flux_regular)
print('Doppler boosted')
Pprint(flux_doppler)

pylab.figure()
pylab.plot(phs, flux_doppler)
pylab.plot(phs, flux_regular)

pylab.figure()
pylab.plot(phs, flux_doppler/flux_regular)

