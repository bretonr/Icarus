# Licensed under a 3-clause BSD style license - see LICENSE

import Icarus
from Icarus.Utils.import_modules import *


##### Welcome message
print( "Analysing some mock data. It is recommended to run it within the `ipython --pylab' environment.\n" )


##### Loading the data
atmo_fln = 'atmo_models.txt'
data_fln = 'data.txt'
ndiv = 5
porb = 10 * 3600
x2sini = 1.1

print( "Loading the data into an Icarus.Photometry object (failure to do so is likely due to missing atmosphere models).\n" )
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


##### Fitting the data using a simple fmin algorithm from scipy
##### Here we make use of the Calc_chi2 function with offset_free = 1 in order to allow for a possible band calibration error, which we assume is 0.3 mag (see column 5 in data.txt).
##### We will also assume that corotation = 1, gravdark = 0.08 and K=300e3.
## Defining the func_par
func_par = lambda p: np.r_[p[0], 1., p[1], p[2], 0.08, 300e3, p[3], p[4], p[5]]
## Wrapper function for the figure of merit to optimize
def FoM(p):
    p = np.asarray(p)
    ## Return large value if parameters are out of bound
    if (p < np.r_[0.1, 0.1, 1500., p[2], 8., 0.]).any() or (p > np.r_[np.pi/2, 1.0, 8000., 8000., 12., 0.1]).any():
        #print( "out-of-bound" )
        return np.ones_like(fit.mag)*1e5
    else:
        chi2, extras = fit.Calc_chi2(p, offset_free=0, func_par=func_par, full_output=True, verbose=False)
        return extras['res']

## Initial guess
par_guess = [70*cts.degree, 0.95, 2000., 5500., 10.3, 0.01]

## Running the fit
print( "Performing a crude fit using the scipy.optimize.leastsq function.\n" )
print( "Beware that the fitting may not converge to the best-fit solution due to local minima. One should try to fit the data using diferent guess parameters or, even better, a more robust fitting algorithm.\n" )
print( "Also, do not expect the best-fit parameter to converge at the actual solution. The reason being that noise is added to the theoretical data when generating the mock data. Hence it might be that by sheer luck the mock data mimic a slightly different set of parameters. If one was to regenerate the mock data several times and rerun the fit, it would on average converge at the actual solution.\n" )
sol = scipy.optimize.leastsq(FoM, par_guess, full_output=True)
par = sol[0]
err = np.sqrt( sol[1].diagonal() )


##### Printing the results
print( "Results from the fitting:" )
print( "{:<28} {:>15}  {:>15}".format("Parameter", "Actual solution", "Fitted Solution") )
print( "{:<28} {:>15.3f}   {:>15.3f} +/- {:.3f}".format("inclination", incl/cts.degree, par[0]/cts.degree, err[0]/cts.degree) )
print( "{:<28} {:>15.3f}   {:>15.3f} +/- {:.3f}".format("filling factor", filling, par[1], err[1]) )
print( "{:<28} {:>15.1f}   {:>15.3f} +/- {:.3f}".format("Tnight", Tnight, par[2], err[2]) )
print( "{:<28} {:>15.1f}   {:>15.3f} +/- {:.3f}".format("Tday", Tday, par[3], err[3]) )
print( "{:<28} {:>15.2f}   {:>15.3f} +/- {:.3f}".format("DM", DM, par[4], err[4]) )
print( "{:<28} {:>15.3f}   {:>15.3f} +/- {:.3f}".format("AJ", AJ, par[5], err[5]) )
print( "" )


##### Plotting the data and model if possible
if pylab:
    print( "Plotting the data. If nothing shows up, try pylab.show()." )
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    pl1 = ax.errorbar(np.r_[fit.data['phase'][0],fit.data['phase'][0]+1.], np.r_[fit.data['mag'][0],fit.data['mag'][0]], yerr=np.r_[fit.data['err'][0],fit.data['err'][0]], marker='s', mfc='red', mec='red', ms=3, ecolor='red', fmt='.')
    pl2 = ax.errorbar(np.r_[fit.data['phase'][1],fit.data['phase'][1]+1.], np.r_[fit.data['mag'][1],fit.data['mag'][1]], yerr=np.r_[fit.data['err'][1],fit.data['err'][1]], marker='s', mfc='blue', mec='blue', ms=3, ecolor='blue', fmt='.')
    phs = np.linspace(0, 2, 101)
    flux = fit.Get_flux_theoretical(par, [phs,phs], func_par=func_par)
    pl3 = ax.plot(phs, flux[0], 'r-')
    pl4 = ax.plot(phs, flux[1], 'b-')
    flux = fit.Get_flux_theoretical(par0, [phs,phs])
    pl5 = ax.plot(phs, flux[0], 'r:')
    pl6 = ax.plot(phs, flux[1], 'b:')
    leg = ax.legend([pl1[0],pl3[0],pl5[0],pl2[0],pl4[0],pl6[0]], ["i","Fit","Real","g","Fit","Real"], ncol=2, loc=0, numpoints=1, scatterpoints=1)
    ax.set_xlabel("Orbital Phase")
    ax.set_ylabel("Magnitude")
    ax.set_xlim([0.,2.])
    vals = np.r_[fit.data['mag'][0], fit.data['mag'][1]]
    ax.set_ylim([vals.max()+(vals.max()-vals.min())*0.1, vals.min()-(vals.max()-vals.min())*0.1])
    pylab.show()




