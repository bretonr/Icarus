# Licensed under a 3-clause BSD style license - see LICENSE

import Icarus
from Icarus.Utils.import_modules import *


##### Welcome message
print( "Generating some mock data." )


##### Making the mock data. The actual mock light curve will be generated later.
phs = np.random.uniform(size=(2, 20))
mag = np.ones(phs.shape, dtype=float)
mag_err = np.abs(np.random.normal(loc=0.2, scale=0.04, size=phs.shape))

np.savetxt('mock_i.txt', np.c_[phs[0], mag[0], mag_err[0]])
np.savetxt('mock_g.txt', np.c_[phs[1], mag[1], mag_err[1]])


##### Loading the data
atmo_fln = 'atmo_models.txt'
data_fln = 'data.txt'
ndiv = 5
porb = 10 * 3600
x2sini = 1.1


##### We create an Icarus photometry instance
fit = Icarus.Photometry.Photometry(atmo_fln, data_fln, ndiv, porb, x2sini)


##### This is the list of true parameters for the stars, as per construction
incl = 75.*cts.degree
corotation = 1.
filling = 0.90
Tnight = 2500.
gravdark = 0.08
K = 300e3
Tday = 5000.
DM = 10.0
AJ = 0.02
par0 = np.r_[incl, corotation, filling, Tnight, gravdark, K, Tday, DM, AJ]


##### Generating theoretical data
mag = fit.Get_flux_theoretical(par0, fit.data['phase'])


##### Adding fake noise
mag[0] = mag[0] + np.random.normal(loc=0.0, scale=0.2, size=mag[0].shape)
mag[1] = mag[1] + np.random.normal(loc=0.0, scale=0.2, size=mag[1].shape)


##### Saving the mock data into the file
np.savetxt('mock_i.txt', np.c_[fit.data['phase'][0], mag[0], fit.data['err'][0]])
np.savetxt('mock_g.txt', np.c_[fit.data['phase'][1], mag[1], fit.data['err'][1]])



