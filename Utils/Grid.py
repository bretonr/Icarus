# Licensed under a 3-clause BSD style license - see LICENSE

from .import_modules import *

logger = logging.getLogger(__name__)


##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##
## Grid utilities
## Contain functions that pertain to the "atmosphere grid-
## related" purposes such as various kinds of interpolation
## in order to extract fluxes.
##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##


def Inter8_photometry(grid, wteff, wlogg, wmu, jteff, jlogg, jmu, area, val_mu):
    """
    """
    code = """
    double fl = 0.;
    #pragma omp parallel shared(grid,wteff,wlogg,wmu,jteff,jlogg,jmu,area,val_mu,nsurf,fl) default(none)
    {
    double w1teff, w0teff, w1logg, w0logg, w1mu, w0mu, tmp_fl;
    int j0teff, j1teff, j0logg, j1logg, j0mu, j1mu;
    #pragma omp for reduction(+:fl)
    for (int i=0; i<nsurf; i++) {
        w1teff = wteff(i);
        w0teff = 1.-w1teff;
        j0teff = jteff(i);
        j1teff = 1.+j0teff;
        w1logg = wlogg(i);
        w0logg = 1.-w1logg;
        j0logg = jlogg(i);
        j1logg = 1.+j0logg;
        w1mu = wmu(i);
        w0mu = 1.-w1mu;
        j0mu = jmu(i);
        j1mu = 1.+j0mu;
        tmp_fl = w1mu*(w0logg*(w0teff*grid(j0teff,j0logg,j1mu) + w1teff*grid(j1teff,j0logg,j1mu)) \
                      + w1logg*(w0teff*grid(j0teff,j1logg,j1mu) + w1teff*grid(j1teff,j1logg,j1mu))) \
                + w0mu*(w0logg*(w0teff*grid(j0teff,j0logg,j0mu) + w1teff*grid(j1teff,j0logg,j0mu)) \
                      + w1logg*(w0teff*grid(j0teff,j1logg,j0mu) + w1teff*grid(j1teff,j1logg,j0mu)));
        fl = fl + exp(tmp_fl) * area(i) * val_mu(i);
    }
    }
    return_val = fl;
    """
    nsurf = jteff.size
    try:
        if os.uname()[0] == 'Darwin':
            extra_compile_args = extra_link_args = ['-O3']
        else:
            extra_compile_args = extra_link_args = ['-O3 -fopenmp']
        get_flux = scipy.weave.inline(code, ['grid', 'wteff', 'wlogg', 'wmu', 'jteff', 'jlogg', 'jmu', 'area', 'val_mu', 'nsurf'], type_converters=scipy.weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=['<omp.h>','<cmath>'], libraries=['m'], verbose=2)
    except:
        get_flux = scipy.weave.inline(code, ['grid', 'wteff', 'wlogg', 'wmu', 'jteff', 'jlogg', 'jmu', 'area', 'val_mu', 'nsurf'], type_converters=scipy.weave.converters.blitz, compiler='gcc', extra_compile_args=['-O3'], extra_link_args=['-O3'], headers=['<cmath>'], libraries=['m'], verbose=2)
    fl = get_flux
    return fl

def Inter8_photometry_details(grid, wteff, wlogg, wmu, jteff, jlogg, jmu, area, val_mu, v, logteff):
    """
    """
    code = """
    double fl = 0.;
    double Keff = 0.;
    double Teff = 0.;
    #pragma omp parallel shared(grid,wteff,wlogg,wmu,jteff,jlogg,jmu,area,val_mu,v,logteff,nsurf,fl,Keff,Teff) default(none)
    {
    double w1teff, w0teff, w1logg, w0logg, w1mu, w0mu, tmp_fl;
    int j0teff, j1teff, j0logg, j1logg, j0mu, j1mu;
    #pragma omp for reduction(+:fl,Keff,Teff)
    for (int i=0; i<nsurf; i++) {
        w1teff = wteff(i);
        w0teff = 1.-w1teff;
        j0teff = jteff(i);
        j1teff = 1.+j0teff;
        w1logg = wlogg(i);
        w0logg = 1.-w1logg;
        j0logg = jlogg(i);
        j1logg = 1.+j0logg;
        w1mu = wmu(i);
        w0mu = 1.-w1mu;
        j0mu = jmu(i);
        j1mu = 1.+j0mu;
        tmp_fl = w1mu*(w0logg*(w0teff*grid(j0teff,j0logg,j1mu) + w1teff*grid(j1teff,j0logg,j1mu)) \
                      + w1logg*(w0teff*grid(j0teff,j1logg,j1mu) + w1teff*grid(j1teff,j1logg,j1mu))) \
                + w0mu*(w0logg*(w0teff*grid(j0teff,j0logg,j0mu) + w1teff*grid(j1teff,j0logg,j0mu)) \
                      + w1logg*(w0teff*grid(j0teff,j1logg,j0mu) + w1teff*grid(j1teff,j1logg,j0mu)));
        tmp_fl = exp(tmp_fl) * area(i) * val_mu(i);
        fl = fl + tmp_fl;
        Keff = Keff + v(i) * tmp_fl;
        Teff = Teff + exp(logteff(i)) * tmp_fl;
    }
    }
    Keff = Keff/fl;
    Teff = Teff/fl;
    results(0) = fl;
    results(1) = Keff;
    results(2) = Teff;
    """
    nsurf = jteff.size
    results = numpy.zeros(3, dtype=float)
    try:
        if os.uname()[0] == 'Darwin':
            extra_compile_args = extra_link_args = ['-O3']
        else:
            extra_compile_args = extra_link_args = ['-O3 -fopenmp']
        get_flux = scipy.weave.inline(code, ['grid', 'wteff', 'wlogg', 'wmu', 'jteff', 'jlogg', 'jmu', 'area', 'val_mu', 'v', 'logteff', 'nsurf', 'results'], type_converters=scipy.weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=['<omp.h>','<cmath>'], libraries=['m'], verbose=2)
    except:
        get_flux = scipy.weave.inline(code, ['grid', 'wteff', 'wlogg', 'wmu', 'jteff', 'jlogg', 'jmu', 'area', 'val_mu', 'v', 'logteff', 'nsurf', 'results'], type_converters=scipy.weave.converters.blitz, compiler='gcc', extra_compile_args=['-O3'], extra_link_args=['-O3'], headers=['<cmath>'], libraries=['m'], verbose=2)
    tmp = get_flux
    fl, Keff, Teff = results
    return fl, Keff, Teff

def Inter8_photometry_Keff(grid, wteff, wlogg, wmu, jteff, jlogg, jmu, area, val_mu, v):
    """
    """
    code = """
    double fl = 0.;
    double Keff = 0.;
    #pragma omp parallel shared(grid,wteff,wlogg,wmu,jteff,jlogg,jmu,area,val_mu,v,nsurf,fl,Keff) default(none)
    {
    double w1teff, w0teff, w1logg, w0logg, w1mu, w0mu, tmp_fl;
    int j0teff, j1teff, j0logg, j1logg, j0mu, j1mu;
    #pragma omp for reduction(+:fl,Keff)
    for (int i=0; i<nsurf; i++) {
        w1teff = wteff(i);
        w0teff = 1.-w1teff;
        j0teff = jteff(i);
        j1teff = 1.+j0teff;
        w1logg = wlogg(i);
        w0logg = 1.-w1logg;
        j0logg = jlogg(i);
        j1logg = 1.+j0logg;
        w1mu = wmu(i);
        w0mu = 1.-w1mu;
        j0mu = jmu(i);
        j1mu = 1.+j0mu;
        tmp_fl = w1mu*(w0logg*(w0teff*grid(j0teff,j0logg,j1mu) + w1teff*grid(j1teff,j0logg,j1mu)) \
                      + w1logg*(w0teff*grid(j0teff,j1logg,j1mu) + w1teff*grid(j1teff,j1logg,j1mu))) \
                + w0mu*(w0logg*(w0teff*grid(j0teff,j0logg,j0mu) + w1teff*grid(j1teff,j0logg,j0mu)) \
                      + w1logg*(w0teff*grid(j0teff,j1logg,j0mu) + w1teff*grid(j1teff,j1logg,j0mu)));
        fl = fl + exp(tmp_fl) * area(i) * val_mu(i);
        Keff = Keff + v(i) * exp(tmp_fl) * area(i) * val_mu(i);
    }
    }
    Keff = Keff/fl;
    results(0) = fl;
    results(1) = Keff;
    """
    nsurf = jteff.size
    results = numpy.zeros(2, dtype=float)
    try:
        if os.uname()[0] == 'Darwin':
            extra_compile_args = extra_link_args = ['-O3']
        else:
            extra_compile_args = extra_link_args = ['-O3 -fopenmp']
        get_flux = scipy.weave.inline(code, ['grid', 'wteff', 'wlogg', 'wmu', 'jteff', 'jlogg', 'jmu', 'area', 'val_mu', 'v', 'nsurf', 'results'], type_converters=scipy.weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=['<omp.h>','<cmath>'], libraries=['m'], verbose=2)
    except:
        get_flux = scipy.weave.inline(code, ['grid', 'wteff', 'wlogg', 'wmu', 'jteff', 'jlogg', 'jmu', 'area', 'val_mu', 'v', 'nsurf', 'results'], type_converters=scipy.weave.converters.blitz, compiler='gcc', extra_compile_args=['-O3'], extra_link_args=['-O3'], headers=['<cmath>'], libraries=['m'], verbose=2)
    tmp = get_flux
    fl, Keff = results
    return fl, Keff

def Inter8_photometry_nosum(grid, wteff, wlogg, wmu, jteff, jlogg, jmu, area, val_mu):
    """
    """
    code = """
    #pragma omp parallel shared(grid,wteff,wlogg,wmu,jteff,jlogg,jmu,area,val_mu,nsurf,fl) default(none)
    {
    double w1teff, w0teff, w1logg, w0logg, w1mu, w0mu, tmp_fl;
    int j0teff, j1teff, j0logg, j1logg, j0mu, j1mu;
    #pragma omp for
    for (int i=0; i<nsurf; i++) {
        w1teff = wteff(i);
        w0teff = 1.-w1teff;
        j0teff = jteff(i);
        j1teff = 1.+j0teff;
        w1logg = wlogg(i);
        w0logg = 1.-w1logg;
        j0logg = jlogg(i);
        j1logg = 1.+j0logg;
        w1mu = wmu(i);
        w0mu = 1.-w1mu;
        j0mu = jmu(i);
        j1mu = 1.+j0mu;
        tmp_fl = w1mu*(w0logg*(w0teff*grid(j0teff,j0logg,j1mu) + w1teff*grid(j1teff,j0logg,j1mu)) \
                      + w1logg*(w0teff*grid(j0teff,j1logg,j1mu) + w1teff*grid(j1teff,j1logg,j1mu))) \
                + w0mu*(w0logg*(w0teff*grid(j0teff,j0logg,j0mu) + w1teff*grid(j1teff,j0logg,j0mu)) \
                      + w1logg*(w0teff*grid(j0teff,j1logg,j0mu) + w1teff*grid(j1teff,j1logg,j0mu)));
        fl(i) = exp(tmp_fl) * area(i) * val_mu(i);
    }
    }
    """
    nsurf = jteff.size
    fl = numpy.zeros(nsurf, dtype=float)
    try:
        if os.uname()[0] == 'Darwin':
            extra_compile_args = extra_link_args = ['-O3']
        else:
            extra_compile_args = extra_link_args = ['-O3 -fopenmp']
        get_flux = scipy.weave.inline(code, ['grid', 'wteff', 'wlogg', 'wmu', 'jteff', 'jlogg', 'jmu', 'area', 'val_mu', 'nsurf', 'fl'], type_converters=scipy.weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=['<omp.h>','<cmath>'], libraries=['m'], verbose=2)
    except:
        get_flux = scipy.weave.inline(code, ['grid', 'wteff', 'wlogg', 'wmu', 'jteff', 'jlogg', 'jmu', 'area', 'val_mu', 'nsurf', 'fl'], type_converters=scipy.weave.converters.blitz, compiler='gcc', extra_compile_args=['-O3'], extra_link_args=['-O3'], headers=['<cmath>'], libraries=['m'], verbose=2)
    tmp = get_flux
    return fl

def Inter8_doppler(grid, wteff, wlogg, wmu, wwav, jteff, jlogg, jmu, jwav, area, val_mu):
    """
    This grid interpolation is made for a grid which is linear in the velocity
    or redshift space, e.g. log lambda.

    Note: Because of the Doppler shift, the interpolation on the wavelength
        will necessarily go out of bound, on the lower or upper range. We 
        assume that the atmosphere grid has a broader spectral coverage than
        the data.
    """
    logger.debug("start")
    grid = numpy.ascontiguousarray(grid)
    wteff = numpy.ascontiguousarray(wteff)
    wlogg = numpy.ascontiguousarray(wlogg)
    wmu = numpy.ascontiguousarray(wmu)
    wwav = numpy.ascontiguousarray(wwav)
    jteff = numpy.ascontiguousarray(jteff)
    jlogg = numpy.ascontiguousarray(jlogg)
    jmu = numpy.ascontiguousarray(jmu)
    jwav = numpy.ascontiguousarray(jwav)
    area = numpy.ascontiguousarray(area)
    val_mu = numpy.ascontiguousarray(val_mu)

    code = """
    #pragma omp parallel shared(grid,wteff,wlogg,wmu,wwav,jteff,jlogg,jmu,jwav,area,val_mu,nsurf,nwav,fl) default(none)
    {
    double w1teff, w0teff, w1logg, w0logg, w1mu, w0mu, w1wav, w0wav;
    int j0teff, j1teff, j0logg, j1logg, j0mu, j1mu, j0wav, j1wav, j0wavk, j1wavk;
    #pragma omp for
    for (int i=0; i<nsurf; i++) {
        w1teff = wteff(i);
        w0teff = 1.-w1teff;
        j0teff = jteff(i);
        j1teff = 1.+j0teff;
        w1logg = wlogg(i);
        w0logg = 1.-w1logg;
        j0logg = jlogg(i);
        j1logg = 1.+j0logg;
        w1mu = wmu(i);
        w0mu = 1.-w1mu;
        j0mu = jmu(i);
        j1mu = 1.+j0mu;
        w1wav = wwav(i);
        w0wav = 1.-w1wav;
        j0wav = jwav(i);
        j1wav = 1.+j0wav;
        //w0wav *= area(i) * val_mu(i);
        //w1wav *= area(i) * val_mu(i);
        for (int k=0; k<nwav; k++) {
            j0wavk = j0wav+k;
            j1wavk = j1wav+k;
            if (j0wavk < 0){
                j0wavk = 0;
                j1wavk = 0;
            } else if (j1wavk >= nwav){
                j0wavk = nwav-1;
                j1wavk = nwav-1;
            }
            fl(k) += \
                ( w1mu * \
                    ( w0wav * \
                        ( w0logg * \
                            ( w0teff * grid(j0teff,j0logg,j1mu,j0wavk) + w1teff * grid(j1teff,j0logg,j1mu,j0wavk) ) \
                        + w1logg * \
                            ( w0teff * grid(j0teff,j1logg,j1mu,j0wavk) + w1teff * grid(j1teff,j1logg,j1mu,j0wavk) ) \
                        ) \
                    + w1wav * \
                        ( w0logg * \
                            ( w0teff * grid(j0teff,j0logg,j1mu,j1wavk) + w1teff * grid(j1teff,j0logg,j1mu,j1wavk) ) \
                        + w1logg * \
                            ( w0teff * grid(j0teff,j1logg,j1mu,j1wavk) + w1teff * grid(j1teff,j1logg,j1mu,j1wavk) ) \
                        ) \
                    ) \
                + w0mu * \
                    ( w0wav * \
                        ( w0logg * \
                            ( w0teff * grid(j0teff,j0logg,j0mu,j0wavk) + w1teff * grid(j1teff,j0logg,j0mu,j0wavk) ) \
                        + w1logg * \
                            ( w0teff * grid(j0teff,j1logg,j0mu,j0wavk) + w1teff * grid(j1teff,j1logg,j0mu,j0wavk) ) \
                        ) \
                    + w1wav * \
                        ( w0logg * \
                            ( w0teff * grid(j0teff,j0logg,j0mu,j1wavk) + w1teff * grid(j1teff,j0logg,j0mu,j1wavk) ) \
                        + w1logg * \
                            ( w0teff * grid(j0teff,j1logg,j0mu,j1wavk) + w1teff * grid(j1teff,j1logg,j0mu,j1wavk) ) \
                        ) \
                    ) \
                ) * area(i) * val_mu(i);
                //);
        }
    }
    }
    """
    nsurf = jteff.size
    nwav = grid.shape[-1]
    fl = numpy.ones(nwav, dtype=float)
    if os.uname()[0] == 'Darwin':
        #extra_compile_args = extra_link_args = ['-O3']
        extra_compile_args = extra_link_args = ['-Ofast']
    else:
        extra_compile_args = extra_link_args = ['-O3 -fopenmp']
    get_flux = scipy.weave.inline(code, ['grid', 'wteff', 'wlogg', 'wmu', 'wwav', 'jteff', 'jlogg', 'jmu', 'jwav', 'area', 'val_mu', 'nsurf', 'nwav', 'fl'], type_converters=scipy.weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=['<omp.h>','<cmath>'], libraries=['m'], verbose=2)
    tmp = get_flux
    logger.debug("start")
    return fl

def Inter8_doppler_savememory(grid, wteff, wlogg, wmu, wwav, jteff, jlogg, jmu, jwav, mu_grid, area, val_mu):
    """
    This grid interpolation is made for a grid which is linear in the velocity
    or redshift space, e.g. log lambda.
    
    Note: Because of the Doppler shift, the interpolation on the wavelength
        will necessarily go out of bound, on the lower or upper range. We 
        assume that the atmosphere grid has a broader spectral coverage than
        the data.
    """
    logger.debug("start")
    grid = numpy.ascontiguousarray(grid)
    wteff = numpy.ascontiguousarray(wteff)
    wlogg = numpy.ascontiguousarray(wlogg)
    wmu = numpy.ascontiguousarray(wmu)
    wwav = numpy.ascontiguousarray(wwav)
    jteff = numpy.ascontiguousarray(jteff)
    jlogg = numpy.ascontiguousarray(jlogg)
    jmu = numpy.ascontiguousarray(jmu)
    jwav = numpy.ascontiguousarray(jwav)
    mu_grid = numpy.ascontiguousarray(mu_grid)
    area = numpy.ascontiguousarray(area)
    val_mu = numpy.ascontiguousarray(val_mu)

    code = """
    #pragma omp parallel shared(grid,wteff,wlogg,wmu,wwav,jteff,jlogg,jmu,jwav,mu_grid,area,val_mu,nsurf,nwav,fl) default(none)
    {
    double w1teff, w0teff, w1logg, w0logg, w1mu, w0mu, w1wav, w0wav;
    int j0teff, j1teff, j0logg, j1logg, j0mu, j1mu, j0wav, j1wav, j0wavk, j1wavk;
    #pragma omp for
    for (int i=0; i<nsurf; i++) {
        w1teff = wteff(i);
        w0teff = 1.-w1teff;
        j0teff = jteff(i);
        j1teff = 1+j0teff;
        w1logg = wlogg(i);
        w0logg = 1.-w1logg;
        j0logg = jlogg(i);
        j1logg = 1+j0logg;
        w1mu = wmu(i);
        w0mu = 1.-w1mu;
        j0mu = jmu(i);
        j1mu = 1+j0mu;
        w1wav = wwav(i);
        w0wav = 1.-w1wav;
        j0wav = jwav(i);
        j1wav = 1+j0wav;
        for (int k=0; k<nwav; k++) {
            j0wavk = j0wav+k;
            j1wavk = j1wav+k;
            if (j0wavk < 0){
                j0wavk = 0;
                j1wavk = 0;
            } else if (j1wavk >= nwav){
                j0wavk = nwav-1;
                j1wavk = nwav-1;
            }
            fl(k) += \
                ( \
                    w0wav * ( w0mu * mu_grid(j0mu,j0wavk) + w1mu * mu_grid(j1mu,j0wavk) ) * \
                        ( \
                        w0logg * ( w0teff * grid(j0teff,j0logg,j0wavk) + w1teff * grid(j1teff,j0logg,j0wavk) ) + \
                        w1logg * ( w0teff * grid(j0teff,j1logg,j0wavk) + w1teff * grid(j1teff,j1logg,j0wavk) ) \
                        ) + \
                    w1wav * ( w0mu * mu_grid(j0mu,j1wavk) + w1mu * mu_grid(j1mu,j1wavk) ) * \
                        ( \
                        w0logg * ( w0teff * grid(j0teff,j0logg,j1wavk) + w1teff * grid(j1teff,j0logg,j1wavk) ) + \
                        w1logg * ( w0teff * grid(j0teff,j1logg,j1wavk) + w1teff * grid(j1teff,j1logg,j1wavk) ) \
                        ) \
                ) * area(i) * val_mu(i);
        }
    }
    }
    """
    nsurf = jteff.size
    nwav = grid.shape[-1]
    mu_grid = numpy.asarray(mu_grid, dtype=float)
    fl = numpy.zeros(nwav, dtype=float)
    if os.uname()[0] == 'Darwin':
        extra_compile_args = extra_link_args = ['-O3']
        headers = ['<cmath>']
    else:
        extra_compile_args = extra_link_args = ['-O3 -fopenmp']
        headers = ['<omp.h>','<cmath>']
    get_flux = scipy.weave.inline(code, ['grid', 'wteff', 'wlogg', 'wmu', 'wwav', 'jteff', 'jlogg', 'jmu', 'jwav','mu_grid', 'area', 'val_mu', 'nsurf', 'nwav', 'fl'], type_converters=scipy.weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=headers, libraries=['m'], verbose=2)
    tmp = get_flux
    logger.debug("end")
    return fl

def Inter8_doppler_savememory_linear(grid, wteff, wlogg, wmu, jteff, jlogg, jmu, mu_grid, area, val_mu, val_vel, z0):
    """
    This grid interpolation is made for a grid which is linear in lambda.
    
    Parameters starting with a 'w' and the weights for interpolation.
    Parameters starting with a 'j' and the indices for interpolation.
    
    teff: temperature
    logg: log gravity
    mu: log emission angle
    
    mu_grid: Grid of intensity for mu and lambda values; shape = n_mu, n_lambda.
    area: Surface area values.
    val_mu: Mu value of each surface area.
    val_vel: Velocity of each surface element.
    z0: delta_lambda / lambda0 of the grid.
        Interpolated lambda bin is: k' = (z+1)*k + z/z0
        Derivation: z+1 = lambda'/lambda
                        = (k'*delta_lambda+lambda0) / (k*delta_lambda+lambda0)
            ... which is solved for n'.
    
    For a surface element, the flux is: flux interpolated * area * mu.
    The interpolated flux takes into account the mu factor and Doppler shift.
    
    Note: Because of the Doppler shift, the interpolation on the wavelength
        will necessarily go out of bound, on the lower or upper range. We 
        assume that the atmosphere grid has a broader spectral coverage than
        the data.
    """
    code = """
    #pragma omp parallel shared(grid,wteff,wlogg,wmu,jteff,jlogg,jmu,mu_grid,area,val_mu,val_vel,z0,nsurf,nlam,fl) default(none)
    {
    double w1teff, w0teff, w1logg, w0logg, w1mu, w0mu;
    int j0teff, j1teff, j0logg, j1logg, j0mu, j1mu;
    double zplusone, kprime, w0k, w1k; // The weights on the interpolated lambda
    int j0k, j1k; // The indices on the interpolated lambda
    #pragma omp for
    for (int k=0; k<nlam; k++) {
        for (int i=0; i<nsurf; i++) {
            // The interpolation on lambda due to the Doppler shift
            zplusone = sqrt( (1.+val_vel(i))/(1.-val_vel(i)) );
            kprime = zplusone*k + (zplusone-1.)/z0;
            if (kprime >= nlam){
                j0k = nlam-1;
                j1k = nlam-1;
                w1k = 1.;
                w0k = 0.;
            } else if (kprime < 0) {
                j0k = 0;
                j1k = 0;
                w1k = 1.;
                w0k = 0.;
            } else {
                j0k = int(kprime);
                j1k = 1+j0k;
                w1k = kprime - j0k;
                w0k = 1.-w1k;
            }
            // Shortcut for the interpolation on other parameters
            w1teff = wteff(i);
            w0teff = 1.-w1teff;
            j0teff = jteff(i);
            j1teff = 1.+j0teff;
            w1logg = wlogg(i);
            w0logg = 1.-w1logg;
            j0logg = jlogg(i);
            j1logg = 1.+j0logg;
            w1mu = wmu(i);
            w0mu = 1.-w1mu;
            j0mu = jmu(i);
            j1mu = 1.+j0mu;
            // We interpolate the grid
            fl(k) += ( \
                (w1mu*mu_grid(j1mu,j0k) + w0mu*mu_grid(j0mu,j0k)) * (w0k*\
                    (w0logg*(w0teff*grid(j0teff,j0logg,j0k) + w1teff*grid(j1teff,j0logg,j0k)) + \
                    w1logg*(w0teff*grid(j0teff,j1logg,j0k) + w1teff*grid(j1teff,j1logg,j0k)))) + \
                (w1mu*mu_grid(j1mu,j1k) + w0mu*mu_grid(j0mu,j1k)) * (w1k*\
                    (w0logg*(w0teff*grid(j0teff,j0logg,j1k) + w1teff*grid(j1teff,j0logg,j1k)) + \
                    w1logg*(w0teff*grid(j0teff,j1logg,j1k) + w1teff*grid(j1teff,j1logg,j1k)))) \
                ) * area(i) * val_mu(i);
        }
    }
    }
    """
    nsurf = jteff.size
    nlam = grid.shape[-1]
    fl = numpy.ones(nlam, dtype=float)
    if os.uname()[0] == 'Darwin':
        extra_compile_args = extra_link_args = ['-O3']
    else:
        extra_compile_args = extra_link_args = ['-O3 -fopenmp']
    get_flux = scipy.weave.inline(code, ['grid', 'wteff', 'wlogg', 'wmu', 'jteff', 'jlogg', 'jmu', 'mu_grid', 'area', 'val_mu', 'val_vel', 'z0', 'nsurf', 'nlam', 'fl'], type_converters=scipy.weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=['<omp.h>','<cmath>'], libraries=['m'], verbose=2)
    tmp = get_flux
    return fl

def Inter8_doppler_nomu(grid, wteff, wlogg, wlam, jteff, jlogg, jlam, area, val_mu):
    """
    Note: Because of the Doppler shift, the interpolation on the wavelength
        will necessarily go out of bound, on the lower or upper range. We 
        assume that the atmosphere grid has a broader spectral coverage than
        the data.
        
        ### Check the fl(k) += foo
    """
    code = """
    #pragma omp parallel shared(grid,wteff,wlogg,wlam,jteff,jlogg,jlam,area,val_mu,nsurf,nlam,fl) default(none)
    {
    double w1teff, w0teff, w1logg, w0logg, w1lam, w0lam;
    int j0teff, j1teff, j0logg, j1logg, j0lam, j1lam, j0lamk, j1lamk;
    #pragma omp for
    for (int i=0; i<nsurf; i++) {
            w1teff = wteff(i);
            w0teff = 1.-w1teff;
            j0teff = jteff(i);
            j1teff = 1.+j0teff;
            w1logg = wlogg(i);
            w0logg = 1.-w1logg;
            j0logg = jlogg(i);
            j1logg = 1.+j0logg;
            w1lam = wlam(i);
            w0lam = 1.-w1lam;
            j0lam = jlam(i);
            j1lam = 1.+j0lam;
        for (int k=0; k<nlam; k++) {
            j0lamk = j0lam+k;
            j1lamk = j1lam+k;
            if (j0lamk < 0){
                j0lamk = 0;
                j1lamk = 0;
            } else if (j1lamk >= nlam){
                j0lamk = nlam-1;
                j1lamk = nlam-1;
            }
            fl(k) += (w0lam*(w0logg*(w0teff*grid(j0teff,j0logg,j0lamk) + w1teff*grid(j1teff,j0logg,j0lamk)) \
                        + w1logg*(w0teff*grid(j0teff,j1logg,j0lamk) + w1teff*grid(j1teff,j1logg,j0lamk))) \
                    + w1lam*(w0logg*(w0teff*grid(j0teff,j0logg,j1lamk) + w1teff*grid(j1teff,j0logg,j1lamk)) \
                        + w1logg*(w0teff*grid(j0teff,j1logg,j1lamk) + w1teff*grid(j1teff,j1logg,j1lamk)))) * area(i) * val_mu(i);
            /*fl(i,k) = pow(10,fl(i,k));*/
        }
    }
    }
    """
    nsurf = jteff.size
    nlam = grid.shape[-1]
    fl = numpy.empty(nlam, dtype=float)
    if os.uname()[0] == 'Darwin':
        extra_compile_args = extra_link_args = ['-O3']
    else:
        extra_compile_args = extra_link_args = ['-O3 -fopenmp']
    get_flux = scipy.weave.inline(code, ['grid', 'wteff', 'wlogg', 'wlam', 'jteff', 'jlogg', 'jlam', 'area', 'val_mu', 'nsurf', 'nlam', 'fl'], type_converters=scipy.weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=['<omp.h>','<cmath>'], libraries=['m'], verbose=2)
    tmp = get_flux
    return fl

def Shift_spectrum(fref, wobs, v, refstart, refstep):
    """Shift_spectrum(fref, wobs0, v, refstart, refstep)
    Takes a reference spectrum, Doppler shifts it, and calculate
    the new spectral flux values at the provided observed wavelengths.
    
    fref: reference flux values
    wobs: observed wavelengths
    v: Doppler velocity shift (in m/s)
    refstart: wavelength of the first reference spectrum data point
    refstep: wavelength step size of the reference spectrum
    
    N.B. Assumes constant bin size for the reference spectrum.
    N.B. Assumes that the observed spectrum bin size is larger than
    the reference spectrum bin size. Otherwise, a simple interpolation
    would be enough.
    N.B. Could be optimized for the case of constant binning for the
    observed spectrum.
    """
    nobs = int(wobs.size)
    nref = int(fref.size)
    fbin = numpy.zeros(nobs, dtype=float)
    fref = numpy.asarray(fref, dtype=float)
    wobs = numpy.asarray(wobs, dtype=float)
    v = float(v)
    refstart = float(refstart)
    refstep = float(refstep)
    code = """
    #line 10
    //double refstart; // start wavelength of the reference spectrum
    //double refstep; // bin width of the reference spectrum
    //int nobs; // length of observed spectrum
    //int nref; // length of reference spectrum
    //double fref; // flux of the reference spectrum
    //double fbin; // integrated flux of the reference spectrum OUTPUT
    //double wobs; // wavelength of the observed spectrum
    double wl, wu; // lower/upper bin limit of the observed spectrum
    double refposl; // index of the lower side of the observed spectrum in the reference spectrum
    double refposu; // index of the upper side of the observed spectrum in the reference spectrum
    int irefl; // rounded integer part of refposl
    int irefu; // rounded integer part of refposu
    double scale = sqrt( (1.+v/299792458.0)/(1.-v/299792458.0) ); // this is the Doppler scaling factor for the observed wavelength
    #line 30
    for (int n=0; n<nobs; ++n) {
        //std::cout << "n: " << n << std::endl;
        if (n == 0) { // special condition for the first data point
            wl = wobs(n) - (wobs(n+1)-wobs(n))*0.5; // the observed bin's lower wavelength value
            wu = (wobs(n)+wobs(n+1))*0.5; // the observed bin's upper wavelength value
            wl *= scale;
            wu *= scale;
        } else if (n < nobs-1) {
            wl = (wobs(n)+wobs(n-1))*0.5; // the observed bin's lower wavelength value
            wu = (wobs(n)+wobs(n+1))*0.5; // the observed bin's upper wavelength value
            wl *= scale;
            wu *= scale;
        } else {
            wl = (wobs(n)+wobs(n-1))*0.5; // the observed bin's lower wavelength value
            wu = wobs(n) + (wobs(n)-wobs(n-1))*0.5; // the observed bin's upper wavelength value
            wl *= scale;
            wu *= scale;
        }
        //std::cout << "wl, wu: " << wl << " " << wu << std::endl;
        #line 50
        refposl = (wl - refstart) / refstep;
        refposu = (wu - refstart) / refstep;
        irefl = (int) (refposl+0.5);
        irefu = (int) (refposu+0.5);
        //std::cout << "refposl, refposu, irefl, irefu: " << refposl << " " << refposu << " " << irefl << " " << irefu << " " << std::endl;
        //std::cout << "fbin(n)1: " << fbin(n) << std::endl;
        if (irefl < 0)
            fbin(n) = fref(0); // assign first flux value if beyond lower reference spectrum limit
        else if (irefu > nref-1)
            fbin(n) = fref(nref-1); // assign last flux value if beyond upper reference spectrum limit
        #line 70
        else {
            if (irefl == irefu) {
                //std::cout << "irefl == irefu" << std::endl;
                fbin(n) += (refposu-refposl) * fref(irefl); // we add fraction of the bin that covers the observed bin
            } else {
                //std::cout << "irefl != irefu" << std::endl;
                fbin(n) += (0.5-(refposl-irefl)) * fref(irefl); // we add the fraction covered by the lower bin of the reference spectrum
                fbin(n) += (0.5+(refposu-irefu)) * fref(irefu); // we add the fraction covered by the upper bin of the reference spectrum
            }
            //std::cout << "fbin(n)2: " << fbin(n) << std::endl;
            for (int i=irefl+1; i<irefu; ++i) {
                fbin(n) += fref(i); // we add the whole bins
            }
            //std::cout << "fbin(n)3: " << fbin(n) << std::endl;
            //if (n == 200) printf( "v: %f, wu-wl: %f, norm: %f\\n", v, (wu-wl), refstep/(wu-wl) );
            fbin(n) *= refstep/(wu-wl); // we normalize in order to get the average flux
            //std::cout << "fbin(n)4: " << fbin(n) << std::endl;
        }
    }
    """
    rebin = scipy.weave.inline(code, ['refstart', 'refstep', 'nobs', 'nref', 'fref', 'fbin', 'wobs', 'v'], type_converters=scipy.weave.converters.blitz, compiler='gcc', libraries=['m'])
    tmp = rebin
    return fbin


