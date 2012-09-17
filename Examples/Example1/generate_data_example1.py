# Licensed under a 3-clause BSD style license - see LICENSE

import Icarus
from Icarus.import_modules import *



##### Welcome message
print( "Generating some mock data." )


##### Making the mock data. The actual mock light curve will be generated later.
phs = numpy.random.uniform(size=(2, 20))
mag = numpy.ones(phs.shape, dtype=float)
mag_err = numpy.abs(numpy.random.normal(loc=0.2, scale=0.04, size=phs.shape))

numpy.savetxt('mock_i.txt', numpy.c_[phs[0], mag[0], mag_err[0]])
numpy.savetxt('mock_g.txt', numpy.c_[phs[1], mag[1], mag_err[1]])


##### Loading the data
atmo_fln = 'atmo_models.txt'
data_fln = 'data.txt'
nalf = 5
porb = 10 * 3600
x2sini = 1.1

incl = 75.*cts.degree
corotation = 1.
filling = 0.90
Tnight = 2500.
gravdark = 0.08
K = 300e3
Tday = 5000.
DM = 10.0
AJ = 0.02
par0 = numpy.r_[incl, corotation, filling, Tnight, gravdark, K, Tday, DM, AJ]

fit = Icarus.Photometry.Photometry(atmo_fln, data_fln, nalf, porb, x2sini)


##### Generating theoretical data
mag = fit.Get_flux_theoretical(par0, fit.data['phase'])


##### Adding fake noise
mag[0] = mag[0] + numpy.random.normal(loc=0.0, scale=0.2, size=mag[0].shape)
mag[1] = mag[1] + numpy.random.normal(loc=0.0, scale=0.2, size=mag[1].shape)


##### Saving the mock data into the file
numpy.savetxt('mock_i.txt', numpy.c_[fit.data['phase'][0], mag[0], fit.data['err'][0]])
numpy.savetxt('mock_g.txt', numpy.c_[fit.data['phase'][1], mag[1], fit.data['err'][1]])


