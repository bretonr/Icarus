# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function, division

import os

try:
    from scipy import weave 
except:
    try:
        import weave
    except:
        print('weave cannot be import from scipy nor on its own.')

from .import_modules import *

logger = logging.getLogger(__name__)


##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##
## Grid utilities
## Contain functions that pertain to the "atmosphere grid-
## related" purposes such as various kinds of interpolation
## in order to extract fluxes.
##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##


def Interp_3Dgrid(grid, wx, wy, wz, jx, jy, jz):
    """
    """
    code = """
    #pragma omp parallel shared(grid,wx,wy,wz,jx,jy,jz,area,val_z,nsurf,interp_val) default(none)
    {
    double w1x, w0x, w1y, w0y, w1z, w0z, tmp_interp_val;
    int j0x, j1x, j0y, j1y, j0z, j1z;
    #pragma omp for reduction(+:fl)
    for (int i=0; i<nsurf; i++) {
        w1x = wx(i);
        w0x = 1.-w1x;
        j0x = jx(i);
        j1x = 1.+j0x;
        w1y = wy(i);
        w0y = 1.-w1y;
        j0y = jy(i);
        j1y = 1.+j0y;
        w1z = wz(i);
        w0z = 1.-w1z;
        j0z = jz(i);
        j1z = 1.+j0z;
        tmp_interp_val = w1z*(w0y*(w0x*grid(j0x,j0y,j1z) + w1x*grid(j1x,j0y,j1z)) \
                      + w1y*(w0x*grid(j0x,j1y,j1z) + w1x*grid(j1x,j1y,j1z))) \
                + w0z*(w0y*(w0x*grid(j0x,j0y,j0z) + w1x*grid(j1x,j0y,j0z)) \
                      + w1y*(w0x*grid(j0x,j1y,j0z) + w1x*grid(j1x,j1y,j0z)));
        interp_val(i) = tmp_interp_val;
    }
    }
    """
    grid = np.ascontiguousarray(grid, dtype=float)
    wx = np.ascontiguousarray(wx, dtype=float)
    wy = np.ascontiguousarray(wy, dtype=float)
    wz = np.ascontiguousarray(wz, dtype=float)
    jx = np.ascontiguousarray(jx, dtype=int)
    jy = np.ascontiguousarray(jy, dtype=int)
    jz = np.ascontiguousarray(jz, dtype=int)
    nsurf = jx.size
    interp_val = np.zeros(nsurf, dtype=float)
    if os.uname()[0] == 'Darwin':
        extra_compile_args = extra_link_args = ['-O3']
        headers = ['<cmath>']
    else:
        extra_compile_args = extra_link_args = ['-O3 -fopenmp']
        headers = ['<omp.h>','<cmath>']
    get_vals = weave.inline(code, ['grid', 'wx', 'wy', 'wz', 'jx', 'jy', 'jz', 'nsurf', 'interp_val'], type_converters=weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=headers, libraries=['m'], verbose=2)
    tmp = get_vals
    return interp_val

def Interp_photometry(grid, wteff, wlogg, wmu, jteff, jlogg, jmu, area, val_mu):
    """
    Simple interpolation of an atmosphere grid having axes (logtemp, logg, mu).

    The interpolation takes a set of points to be interpolated and summed
    together.

    Parameters
    ----------
    grid : ndarray
        Atmosphere grid, with dimensions (logtemp, logg, mu, wav).
    wteff, wlogg, wmu : ndarray
        Weights of the temperature, logg, mu.
    jteff, jlogg, jmu : ndarray
        Fractional position of the temperature, logg, mu.
    area : ndarray
        Area (i.e. weight) of each surface element for the summation.
    val_mu : ndarray
        Value of the cross-section visible to us.

    Returns
    -------
    flux : scalar
        Flux integrated over the surface.
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
    grid = np.ascontiguousarray(grid, dtype=float)
    wteff = np.ascontiguousarray(wteff, dtype=float)
    wlogg = np.ascontiguousarray(wlogg, dtype=float)
    wmu = np.ascontiguousarray(wmu, dtype=float)
    jteff = np.ascontiguousarray(jteff, dtype=int)
    jlogg = np.ascontiguousarray(jlogg, dtype=int)
    jmu = np.ascontiguousarray(jmu, dtype=int)
    area = np.ascontiguousarray(area, dtype=float)
    val_mu = np.ascontiguousarray(val_mu, dtype=float)
    nsurf = jteff.size
    if os.uname()[0] == 'Darwin':
        extra_compile_args = extra_link_args = ['-O3']
        headers = ['<cmath>']
    else:
        extra_compile_args = extra_link_args = ['-O3 -fopenmp']
        headers = ['<omp.h>','<cmath>']
    get_flux = weave.inline(code, ['grid', 'wteff', 'wlogg', 'wmu', 'jteff', 'jlogg', 'jmu', 'area', 'val_mu', 'nsurf'], type_converters=weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=headers, libraries=['m'], verbose=2)
    fl = get_flux
    return fl

def Interp_photometry_doppler(grid, wteff, wlogg, wmu, jteff, jlogg, jmu, area, val_mu, val_vel, grid_doppler):
    """
    Simple interpolation of an atmosphere grid having axes (logtemp, logg, mu),
    which also takes into account Doppler boosting using coefficients stored in
    a dedicated grid.

    Parameters
    ----------
    The interpolation takes a set of points to be interpolated and summed together.
    grid : ndarray
        Atmosphere grid, with dimensions (logtemp, logg, mu, wav).
    wteff, wlogg, wmu : ndarray
        Weights of the temperature, logg, mu.
    jteff, jlogg, jmu : ndarray
        Fractional position of the temperature, logg, mu.
    area : ndarray
        Area (i.e. weight) of each surface element for the summation.
    val_mu : ndarray
        Value of the cross-section visible to us.
    val_vel : ndarray
        Value of the velocity, in v/c units.
    grid_doppler : ndarray
        Doppler boosting coefficients, with dimensions similar to grid.

    Returns
    -------
    flux : scalar
        Flux integrated over the surface, with Doppler boosting.
    """
    code = """
    double fl = 0.;
    #pragma omp parallel shared(grid,wteff,wlogg,wmu,jteff,jlogg,jmu,area,val_mu,nsurf,val_vel,grid_doppler,fl) default(none)
    {
    double w1teff, w0teff, w1logg, w0logg, w1mu, w0mu, tmp_fl, tmp_doppler;
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
        tmp_doppler = w1mu*(w0logg*(w0teff*grid_doppler(j0teff,j0logg,j1mu) + w1teff*grid_doppler(j1teff,j0logg,j1mu)) \
                      + w1logg*(w0teff*grid_doppler(j0teff,j1logg,j1mu) + w1teff*grid_doppler(j1teff,j1logg,j1mu))) \
                + w0mu*(w0logg*(w0teff*grid_doppler(j0teff,j0logg,j0mu) + w1teff*grid_doppler(j1teff,j0logg,j0mu)) \
                      + w1logg*(w0teff*grid_doppler(j0teff,j1logg,j0mu) + w1teff*grid_doppler(j1teff,j1logg,j0mu)));
        fl = fl + exp(tmp_fl) * area(i) * val_mu(i) * (1 + tmp_doppler * val_vel(i));
    }
    }
    return_val = fl;
    """
    grid = np.ascontiguousarray(grid, dtype=float)
    wteff = np.ascontiguousarray(wteff, dtype=float)
    wlogg = np.ascontiguousarray(wlogg, dtype=float)
    wmu = np.ascontiguousarray(wmu, dtype=float)
    jteff = np.ascontiguousarray(jteff, dtype=int)
    jlogg = np.ascontiguousarray(jlogg, dtype=int)
    jmu = np.ascontiguousarray(jmu, dtype=int)
    area = np.ascontiguousarray(area, dtype=float)
    val_mu = np.ascontiguousarray(val_mu, dtype=float)
    grid_doppler = np.ascontiguousarray(grid_doppler, dtype=float)
    val_vel = np.ascontiguousarray(val_vel, dtype=float)
    nsurf = jteff.size
    if os.uname()[0] == 'Darwin':
        extra_compile_args = extra_link_args = ['-O3']
        headers = ['<cmath>']
    else:
        extra_compile_args = extra_link_args = ['-O3 -fopenmp']
        headers = ['<omp.h>','<cmath>']
    get_flux = weave.inline(code, ['grid', 'wteff', 'wlogg', 'wmu', 'jteff', 'jlogg', 'jmu', 'area', 'val_mu', 'nsurf', 'val_vel', 'grid_doppler'], type_converters=weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=headers, libraries=['m'], verbose=2)
    fl = get_flux
    return fl

def Interp_photometry_doppler_nosum(grid, wteff, wlogg, wmu, jteff, jlogg, jmu, area, val_mu, val_vel, grid_doppler):
    """
    Simple interpolation of an atmosphere grid having axes (logtemp, logg, mu),
    which also takes into account Doppler boosting using coefficients stored in
    a dedicated grid.

    Note: As opposed to Interp_photometry_doppler, this function does not sum
    the surface elements.

    Parameters
    ----------
    The interpolation takes a set of points to be interpolated.
    grid : ndarray
        Atmosphere grid, with dimensions (logtemp, logg, mu, wav).
    wteff, wlogg, wmu : ndarray
        Weights of the temperature, logg, mu.
    jteff, jlogg, jmu : ndarray
        Fractional position of the temperature, logg, mu.
    area : ndarray
        Area (i.e. weight) of each surface element for the summation.
    val_mu : ndarray
        Value of the cross-section visible to us.
    val_vel : ndarray
        Value of the velocity, in v/c units.
    grid_doppler : ndarray
        Doppler boosting coefficients, with dimensions similar to grid.

    Returns
    -------
    flux : ndarray
        Flux _not_ integrated over the surface.
    """
    code = """
    #pragma omp parallel shared(grid,wteff,wlogg,wmu,jteff,jlogg,jmu,area,val_mu,nsurf,val_vel,grid_doppler,fl) default(none)
    {
    double w1teff, w0teff, w1logg, w0logg, w1mu, w0mu, tmp_fl, tmp_doppler;
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
        tmp_doppler = w1mu*(w0logg*(w0teff*grid_doppler(j0teff,j0logg,j1mu) + w1teff*grid_doppler(j1teff,j0logg,j1mu)) \
                      + w1logg*(w0teff*grid_doppler(j0teff,j1logg,j1mu) + w1teff*grid_doppler(j1teff,j1logg,j1mu))) \
                + w0mu*(w0logg*(w0teff*grid_doppler(j0teff,j0logg,j0mu) + w1teff*grid_doppler(j1teff,j0logg,j0mu)) \
                      + w1logg*(w0teff*grid_doppler(j0teff,j1logg,j0mu) + w1teff*grid_doppler(j1teff,j1logg,j0mu)));
        fl(i) = exp(tmp_fl) * area(i) * val_mu(i) * (1 + tmp_doppler * val_vel(i));
    }
    }
    """
    grid = np.ascontiguousarray(grid, dtype=float)
    wteff = np.ascontiguousarray(wteff, dtype=float)
    wlogg = np.ascontiguousarray(wlogg, dtype=float)
    wmu = np.ascontiguousarray(wmu, dtype=float)
    jteff = np.ascontiguousarray(jteff, dtype=int)
    jlogg = np.ascontiguousarray(jlogg, dtype=int)
    jmu = np.ascontiguousarray(jmu, dtype=int)
    area = np.ascontiguousarray(area, dtype=float)
    val_mu = np.ascontiguousarray(val_mu, dtype=float)
    grid_doppler = np.ascontiguousarray(grid_doppler, dtype=float)
    val_vel = np.ascontiguousarray(val_vel, dtype=float)
    nsurf = jteff.size
    fl = np.zeros(nsurf, dtype=float)
    if os.uname()[0] == 'Darwin':
        extra_compile_args = extra_link_args = ['-O3']
        headers = ['<cmath>']
    else:
        extra_compile_args = extra_link_args = ['-O3 -fopenmp']
        headers = ['<omp.h>','<cmath>']
    get_flux = weave.inline(code, ['grid', 'wteff', 'wlogg', 'wmu', 'jteff', 'jlogg', 'jmu', 'area', 'val_mu', 'nsurf', 'val_vel', 'grid_doppler', 'fl'], type_converters=weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=headers, libraries=['m'], verbose=2)
    tmp = get_flux
    return fl

def Interp_photometry_details(grid, wteff, wlogg, wmu, jteff, jlogg, jmu, area, val_mu, v, val_teff):
    """
    Simple interpolation of an atmosphere grid having axes (logtemp, logg, mu),
    which also takes into account Doppler boosting using coefficients stored in
    a dedicated grid.

    Note: Some additional quantities are calculated, such as the flux-weighted
    velocity, temperature and vsini.

    Parameters
    ----------
    The interpolation takes a set of points to be interpolated and summed together.
    grid : ndarray
        Atmosphere grid, with dimensions (logtemp, logg, mu, wav).
    wteff, wlogg, wmu : ndarray
        Weights of the temperature, logg, mu.
    jteff, jlogg, jmu : ndarray
        Fractional position of the temperature, logg, mu.
    area : ndarray
        Area (i.e. weight) of each surface element for the summation.
    val_mu : ndarray
        Value of the cross-section visible to us.
    val_vel : ndarray
        Value of the velocity, in v/c units.
    val_teff : ndarray
        Value of the temperatures.

    Returns
    -------
    flux : scalar
        Flux integrated over the surface.
    Keff : scalar
        Flux-weighted radial velocity.
    vsini : scalar
        Estimated vsini.
    Teff : scalar
        Flux-weighted temperature.
    """
    code = """
    double fl = 0.;
    double Keff = 0.;
    double KeffSquare = 0;
    double Teff = 0.;
    double vsini = 0.;
    #pragma omp parallel shared(grid,wteff,wlogg,wmu,jteff,jlogg,jmu,area,val_mu,v,val_teff,nsurf,fl,Keff,KeffSquare,Teff) default(none)
    {
    double w1teff, w0teff, w1logg, w0logg, w1mu, w0mu, tmp_fl;
    int j0teff, j1teff, j0logg, j1logg, j0mu, j1mu;
    #pragma omp for reduction(+:fl,Keff,KeffSquare,Teff)
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
        KeffSquare = KeffSquare + v(i)*v(i) * tmp_fl;
        Teff = Teff + exp(val_teff(i)) * tmp_fl;
    }
    }
    Keff = Keff/fl;
    Teff = Teff/fl;
    vsini = sqrt((KeffSquare/fl) - Keff*Keff);
    results(0) = fl;
    results(1) = Keff;
    results(2) = vsini;
    results(3) = Teff;
    """
    grid = np.ascontiguousarray(grid, dtype=float)
    wteff = np.ascontiguousarray(wteff, dtype=float)
    wlogg = np.ascontiguousarray(wlogg, dtype=float)
    wmu = np.ascontiguousarray(wmu, dtype=float)
    jteff = np.ascontiguousarray(jteff, dtype=int)
    jlogg = np.ascontiguousarray(jlogg, dtype=int)
    jmu = np.ascontiguousarray(jmu, dtype=int)
    area = np.ascontiguousarray(area, dtype=float)
    val_mu = np.ascontiguousarray(val_mu, dtype=float)
    v = np.ascontiguousarray(v, dtype=float)
    val_teff = np.ascontiguousarray(val_teff, dtype=float)
    nsurf = jteff.size
    results = np.zeros(4, dtype=float)
    if os.uname()[0] == 'Darwin':
        extra_compile_args = extra_link_args = ['-O3']
        headers = ['<cmath>']
    else:
        extra_compile_args = extra_link_args = ['-O3 -fopenmp']
        headers = ['<omp.h>','<cmath>']
    get_flux = weave.inline(code, ['grid', 'wteff', 'wlogg', 'wmu', 'jteff', 'jlogg', 'jmu', 'area', 'val_mu', 'v', 'val_teff', 'nsurf', 'results'], type_converters=weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=headers, libraries=['m'], verbose=2)
    tmp = get_flux
    fl, Keff, vsini, Teff = results
    return fl, Keff, vsini, Teff

def Interp_photometry_Keff(grid, wteff, wlogg, wmu, jteff, jlogg, jmu, area, val_mu, v):
    """
    Simple interpolation of an atmosphere grid having axes (logtemp, logg, mu),
    which also takes into account Doppler boosting using coefficients stored in
    a dedicated grid.

    Note: The flux-weighted velocity is also returned.

    Parameters
    ----------
    The interpolation takes a set of points to be interpolated and summed together.
    grid : ndarray
        Atmosphere grid, with dimensions (logtemp, logg, mu, wav).
    wteff, wlogg, wmu : ndarray
        Weights of the temperature, logg, mu.
    jteff, jlogg, jmu : ndarray
        Fractional position of the temperature, logg, mu.
    area : ndarray
        Area (i.e. weight) of each surface element for the summation.
    val_mu : ndarray
        Value of the cross-section visible to us.
    val_vel : ndarray
        Value of the velocity, in v/c units.

    Returns
    -------
    flux : scalar
        Flux integrated over the surface.
    Keff : scalar
        Flux-weighted radial velocity.
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
    grid = np.ascontiguousarray(grid, dtype=float)
    wteff = np.ascontiguousarray(wteff, dtype=float)
    wlogg = np.ascontiguousarray(wlogg, dtype=float)
    wmu = np.ascontiguousarray(wmu, dtype=float)
    jteff = np.ascontiguousarray(jteff, dtype=int)
    jlogg = np.ascontiguousarray(jlogg, dtype=int)
    jmu = np.ascontiguousarray(jmu, dtype=int)
    area = np.ascontiguousarray(area, dtype=float)
    val_mu = np.ascontiguousarray(val_mu, dtype=float)
    v = np.ascontiguousarray(v, dtype=float)
    nsurf = jteff.size
    results = np.zeros(2, dtype=float)
    if os.uname()[0] == 'Darwin':
        extra_compile_args = extra_link_args = ['-O3']
        headers = ['<cmath>']
    else:
        extra_compile_args = extra_link_args = ['-O3 -fopenmp']
        headers = ['<omp.h>','<cmath>']
    get_flux = weave.inline(code, ['grid', 'wteff', 'wlogg', 'wmu', 'jteff', 'jlogg', 'jmu', 'area', 'val_mu', 'v', 'nsurf', 'results'], type_converters=weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=headers, libraries=['m'], verbose=2)
    tmp = get_flux
    fl, Keff = results
    return fl, Keff

def Interp_photometry_nosum(grid, wteff, wlogg, wmu, jteff, jlogg, jmu, area, val_mu):
    """
    Simple interpolation of an atmosphere grid having axes (logtemp, logg, mu).

    Note: As opposed to Interp_photometry, this function does not sum
    the surface elements.

    Parameters
    ----------
    The interpolation takes a set of points to be interpolated.
    grid : ndarray
        Atmosphere grid, with dimensions (logtemp, logg, mu, wav).
    wteff, wlogg, wmu : ndarray
        Weights of the temperature, logg, mu.
    jteff, jlogg, jmu : ndarray
        Fractional position of the temperature, logg, mu.
    area : ndarray
        Area (i.e. weight) of each surface element for the summation.
    val_mu : ndarray
        Value of the cross-section visible to us.

    Returns
    -------
    flux : ndarray
        Flux _not_ integrated over the surface.
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
    grid = np.ascontiguousarray(grid, dtype=float)
    wteff = np.ascontiguousarray(wteff, dtype=float)
    wlogg = np.ascontiguousarray(wlogg, dtype=float)
    wmu = np.ascontiguousarray(wmu, dtype=float)
    jteff = np.ascontiguousarray(jteff, dtype=int)
    jlogg = np.ascontiguousarray(jlogg, dtype=int)
    jmu = np.ascontiguousarray(jmu, dtype=int)
    area = np.ascontiguousarray(area, dtype=float)
    val_mu = np.ascontiguousarray(val_mu, dtype=float)
    nsurf = jteff.size
    fl = np.zeros(nsurf, dtype=float)
    if os.uname()[0] == 'Darwin':
        extra_compile_args = extra_link_args = ['-O3']
        headers = ['<cmath>']
    else:
        extra_compile_args = extra_link_args = ['-O3 -fopenmp']
        headers = ['<omp.h>','<cmath>']
    get_flux = weave.inline(code, ['grid', 'wteff', 'wlogg', 'wmu', 'jteff', 'jlogg', 'jmu', 'area', 'val_mu', 'nsurf', 'fl'], type_converters=weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=headers, libraries=['m'], verbose=2)
    tmp = get_flux
    return fl

def Interp_doppler(grid, wteff, wlogg, wmu, wwav, jteff, jlogg, jmu, jwav, area, val_mu):
    """
    Simple interpolation of an atmosphere grid having axes (logtemp, logg, mu, wav).

    This grid interpolation is made for a grid which is linear in the velocity
    or redshift space, e.g. log lambda.

    Note: Because of the Doppler shift, the interpolation on the wavelength
        will necessarily go out of bound, on the lower or upper range. We
        assume that the atmosphere grid has a broader spectral coverage than
        the data.

    Parameters
    ----------
    The interpolation takes a set of points to be interpolated.
    grid : ndarray
        Atmosphere grid, with dimensions (logtemp, logg, mu, wav).
    wteff, wlogg, wmu, wwav : ndarray
        Weights of the temperature, logg, mu, wav.
    jteff, jlogg, jmu, jwav : ndarray
        Fractional position of the temperature, logg, mu, wav.
    area : ndarray
        Area (i.e. weight) of each surface element for the summation.
    val_mu : ndarray
        Value of the cross-section visible to us.

    Returns
    -------
    spectrum : ndarray
        Spectrum integrated over the surface.
    """
    logger.log(9, "start")
    code = """
    #pragma omp parallel shared(grid,wteff,wlogg,wmu,wwav,jteff,jlogg,jmu,jwav,area,val_mu,nsurf,nwav,fl) default(none)
    {
    double w1teff, w0teff, w1logg, w0logg, w1mu, w0mu, w1wav, w0wav, tmp_fl;
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
            tmp_fl = \
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
                );
            //std::cout << "tmp_fl " << tmp_fl << std::endl;
            //std::cout << "area*val_mu " << area(i) * val_mu(i) << std::endl;
            //std::cout << "fl " << tmp_fl * area(i) * val_mu(i) << std::endl;
            //fl(k) += tmp_fl * area(i) * val_mu(i);
            fl(k) += exp(tmp_fl) * area(i) * val_mu(i);
                //);
        }
    }
    }
    """
    grid = np.ascontiguousarray(grid, dtype=float)
    wteff = np.ascontiguousarray(wteff, dtype=float)
    wlogg = np.ascontiguousarray(wlogg, dtype=float)
    wmu = np.ascontiguousarray(wmu, dtype=float)
    jteff = np.ascontiguousarray(jteff, dtype=int)
    jlogg = np.ascontiguousarray(jlogg, dtype=int)
    jmu = np.ascontiguousarray(jmu, dtype=int)
    area = np.ascontiguousarray(area, dtype=float)
    val_mu = np.ascontiguousarray(val_mu, dtype=float)
    wwav = np.ascontiguousarray(wwav, dtype=float)
    jwav = np.ascontiguousarray(jwav, dtype=int)
    nsurf = jteff.size
    nwav = grid.shape[-1]
    fl = np.zeros(nwav, dtype=float)
    if os.uname()[0] == 'Darwin':
        extra_compile_args = extra_link_args = ['-O3']
        headers = ['<cmath>']
    else:
        extra_compile_args = extra_link_args = ['-O3 -fopenmp']
        headers = ['<omp.h>','<cmath>']
    get_flux = weave.inline(code, ['grid', 'wteff', 'wlogg', 'wmu', 'wwav', 'jteff', 'jlogg', 'jmu', 'jwav', 'area', 'val_mu', 'nsurf', 'nwav', 'fl'], type_converters=weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=headers, libraries=['m'], verbose=2)
    tmp = get_flux
    logger.log(9, "end")
    return fl

def Interp_doppler_savememory(grid, wteff, wlogg, wmu, wwav, jteff, jlogg, jmu, jwav, mu_grid, area, val_mu):
    """
    Simple interpolation of an atmosphere grid having axes (logtemp, logg, wav).

    This grid interpolation is made for a grid which is linear in the velocity
    or redshift space, e.g. log lambda.

    The limb darkening is implement by sourcing values from an external grid
    containing limb darkening coefficients.

    Note: Because of the Doppler shift, the interpolation on the wavelength
        will necessarily go out of bound, on the lower or upper range. We
        assume that the atmosphere grid has a broader spectral coverage than
        the data.

    Parameters
    ----------
    The interpolation takes a set of points to be interpolated.
    grid : ndarray
        Atmosphere grid, with dimensions (logtemp, logg, mu, wav).
    wteff, wlogg, wmu, wwav : ndarray
        Weights of the temperature, logg, mu, wav.
    jteff, jlogg, jmu, jwav : ndarray
        Fractional position of the temperature, logg, mu, wav.
    area : ndarray
        Area (i.e. weight) of each surface element for the summation.
    val_mu : ndarray
        Value of the cross-section visible to us.
    mu_grid : ndarray
        Grid of limb darkening having axes (mu, wav).

    Returns
    -------
    spectrum : ndarray
        Spectrum integrated over the surface.

    NOTE: This is becoming obsolete.
    """
    logger.log(9, "start")
    code = """
    #pragma omp parallel shared(grid,wteff,wlogg,wmu,wwav,jteff,jlogg,jmu,jwav,mu_grid,area,val_mu,nsurf,nwav,fl) default(none)
    {
    double w1teff, w0teff, w1logg, w0logg, w1mu, w0mu, w1wav, w0wav, tmp_fl;
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
            tmp_fl = \
                ( \
                    w0wav * ( w0mu * mu_grid(j0mu,j0wavk) + w1mu * mu_grid(j1mu,j0wavk) ) * \
                        exp( \
                        w0logg * ( w0teff * grid(j0teff,j0logg,j0wavk) + w1teff * grid(j1teff,j0logg,j0wavk) ) + \
                        w1logg * ( w0teff * grid(j0teff,j1logg,j0wavk) + w1teff * grid(j1teff,j1logg,j0wavk) ) \
                        ) + \
                    w1wav * ( w0mu * mu_grid(j0mu,j1wavk) + w1mu * mu_grid(j1mu,j1wavk) ) * \
                        exp( \
                        w0logg * ( w0teff * grid(j0teff,j0logg,j1wavk) + w1teff * grid(j1teff,j0logg,j1wavk) ) + \
                        w1logg * ( w0teff * grid(j0teff,j1logg,j1wavk) + w1teff * grid(j1teff,j1logg,j1wavk) ) \
                        ) \
                );
            fl(k) += tmp_fl * area(i) * val_mu(i);
        }
    }
    }
    """
    grid = np.ascontiguousarray(grid, dtype=float)
    wteff = np.ascontiguousarray(wteff, dtype=float)
    wlogg = np.ascontiguousarray(wlogg, dtype=float)
    wmu = np.ascontiguousarray(wmu, dtype=float)
    jteff = np.ascontiguousarray(jteff, dtype=int)
    jlogg = np.ascontiguousarray(jlogg, dtype=int)
    jmu = np.ascontiguousarray(jmu, dtype=int)
    area = np.ascontiguousarray(area, dtype=float)
    val_mu = np.ascontiguousarray(val_mu, dtype=float)
    wwav = np.ascontiguousarray(wwav, dtype=float)
    jwav = np.ascontiguousarray(jwav, dtype=int)
    mu_grid = np.ascontiguousarray(mu_grid, dtype=float)
    nsurf = jteff.size
    nwav = grid.shape[-1]
    fl = np.zeros(nwav, dtype=float)
    if os.uname()[0] == 'Darwin':
        extra_compile_args = extra_link_args = ['-O3']
        headers = ['<cmath>']
    else:
        extra_compile_args = extra_link_args = ['-O3 -fopenmp']
        headers = ['<omp.h>','<cmath>']
    get_flux = weave.inline(code, ['grid', 'wteff', 'wlogg', 'wmu', 'wwav', 'jteff', 'jlogg', 'jmu', 'jwav','mu_grid', 'area', 'val_mu', 'nsurf', 'nwav', 'fl'], type_converters=weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=headers, libraries=['m'], verbose=2)
    tmp = get_flux
    logger.log(9, "end")
    return fl

def Interp_doppler_savememory_linear(grid, wteff, wlogg, wmu, jteff, jlogg, jmu, mu_grid, area, val_mu, val_vel, z0):
    """
    Simple interpolation of an atmosphere grid having axes (logtemp, logg, wav).

    This grid interpolation is made for a grid which is linear in lambda space.

    The limb darkening is implement by sourcing values from an external grid
    containing limb darkening coefficients.

    Note: Because of the Doppler shift, the interpolation on the wavelength
        will necessarily go out of bound, on the lower or upper range. We
        assume that the atmosphere grid has a broader spectral coverage than
        the data.

    Parameters
    ----------
    The interpolation takes a set of points to be interpolated.
    grid : ndarray
        Atmosphere grid, with dimensions (logtemp, logg, mu, wav).
    wteff, wlogg, wmu, wwav : ndarray
        Weights of the temperature, logg, mu, wav.
    jteff, jlogg, jmu, jwav : ndarray
        Fractional position of the temperature, logg, mu, wav.
    area : ndarray
        Area (i.e. weight) of each surface element for the summation.
    val_mu : ndarray
        Value of the cross-section visible to us.
    mu_grid : ndarray
        Grid of limb darkening having axes (mu, wav).
    z0: delta_lambda / lambda0 of the grid.
        Interpolated lambda bin is: k' = (z+1)*k + z/z0
        Derivation: z+1 = lambda'/lambda
                        = (k'*delta_lambda+lambda0) / (k*delta_lambda+lambda0)
            ... which is solved for n'.

    Returns
    -------
    spectrum : ndarray
        Spectrum integrated over the surface.

    NOTE: This is becoming obsolete.
    """
    code = """
    #pragma omp parallel shared(grid,wteff,wlogg,wmu,jteff,jlogg,jmu,mu_grid,area,val_mu,val_vel,z0,nsurf,nwav,fl) default(none)
    {
    double w1teff, w0teff, w1logg, w0logg, w1mu, w0mu;
    int j0teff, j1teff, j0logg, j1logg, j0mu, j1mu;
    double zplusone, kprime, w0k, w1k; // The weights on the interpolated lambda
    int j0k, j1k; // The indices on the interpolated lambda
    #pragma omp for
    for (int k=0; k<nwav; k++) {
        for (int i=0; i<nsurf; i++) {
            // The interpolation on lambda due to the Doppler shift
            zplusone = sqrt( (1.+val_vel(i))/(1.-val_vel(i)) );
            kprime = zplusone*k + (zplusone-1.)/z0;
            if (kprime >= nwav){
                j0k = nwav-1;
                j1k = nwav-1;
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
    grid = np.ascontiguousarray(grid, dtype=float)
    wteff = np.ascontiguousarray(wteff, dtype=float)
    wlogg = np.ascontiguousarray(wlogg, dtype=float)
    wmu = np.ascontiguousarray(wmu, dtype=float)
    jteff = np.ascontiguousarray(jteff, dtype=int)
    jlogg = np.ascontiguousarray(jlogg, dtype=int)
    jmu = np.ascontiguousarray(jmu, dtype=int)
    area = np.ascontiguousarray(area, dtype=float)
    val_mu = np.ascontiguousarray(val_mu, dtype=float)
    wwav = np.ascontiguousarray(wwav, dtype=float)
    jwav = np.ascontiguousarray(jwav, dtype=int)
    nsurf = jteff.size
    nwav = grid.shape[-1]
    fl = np.zeros(nwav, dtype=float)
    if os.uname()[0] == 'Darwin':
        extra_compile_args = extra_link_args = ['-O3']
        headers = ['<cmath>']
    else:
        extra_compile_args = extra_link_args = ['-O3 -fopenmp']
        headers = ['<omp.h>','<cmath>']
    get_flux = weave.inline(code, ['grid', 'wteff', 'wlogg', 'wmu', 'jteff', 'jlogg', 'jmu', 'mu_grid', 'area', 'val_mu', 'val_vel', 'z0', 'nsurf', 'nwav', 'fl'], type_converters=weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=headers, libraries=['m'], verbose=2)
    tmp = get_flux
    return fl

def Interp_doppler_nomu(grid, wteff, wlogg, wwav, jteff, jlogg, jwav, area, val_mu):
    """
    Simple interpolation of an atmosphere grid having axes (logtemp, logg, wav).

    This grid interpolation is made for a grid which is linear in the velocity
    or redshift space, e.g. log lambda.

    Note: Because of the Doppler shift, the interpolation on the wavelength
        will necessarily go out of bound, on the lower or upper range. We
        assume that the atmosphere grid has a broader spectral coverage than
        the data.

    Parameters
    ----------
    The interpolation takes a set of points to be interpolated.
    grid : ndarray
        Atmosphere grid, with dimensions (logtemp, logg, mu, wav).
    wteff, wlogg, wmu, wwav : ndarray
        Weights of the temperature, logg, mu, wav.
    jteff, jlogg, jmu, jwav : ndarray
        Fractional position of the temperature, logg, mu, wav.
    area : ndarray
        Area (i.e. weight) of each surface element for the summation.
    val_mu : ndarray
        Value of the cross-section visible to us.
    mu_grid : ndarray
        Grid of limb darkening having axes (mu, wav).
    z0: delta_lambda / lambda0 of the grid.
        Interpolated lambda bin is: k' = (z+1)*k + z/z0
        Derivation: z+1 = lambda'/lambda
                        = (k'*delta_lambda+lambda0) / (k*delta_lambda+lambda0)
            ... which is solved for n'.

    Returns
    -------
    spectrum : ndarray
        Spectrum integrated over the surface.
    """
    code = """
    #pragma omp parallel shared(grid,wteff,wlogg,wwav,jteff,jlogg,jwav,area,val_mu,nsurf,nwav,fl) default(none)
    {
    double w1teff, w0teff, w1logg, w0logg, w1wav, w0wav;
    int j0teff, j1teff, j0logg, j1logg, j0wav, j1wav, j0wavk, j1wavk;
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
        w1wav = wwav(i);
        w0wav = 1.-w1wav;
        j0wav = jwav(i);
        j1wav = 1.+j0wav;
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
            fl(k) += (w0wav*(w0logg*(w0teff*grid(j0teff,j0logg,j0wavk) + w1teff*grid(j1teff,j0logg,j0wavk)) \
                        + w1logg*(w0teff*grid(j0teff,j1logg,j0wavk) + w1teff*grid(j1teff,j1logg,j0wavk))) \
                    + w1wav*(w0logg*(w0teff*grid(j0teff,j0logg,j1wavk) + w1teff*grid(j1teff,j0logg,j1wavk)) \
                        + w1logg*(w0teff*grid(j0teff,j1logg,j1wavk) + w1teff*grid(j1teff,j1logg,j1wavk)))) * area(i) * val_mu(i);
            /*fl(i,k) = pow(10,fl(i,k));*/
        }
    }
    }
    """
    grid = np.ascontiguousarray(grid, dtype=float)
    wteff = np.ascontiguousarray(wteff, dtype=float)
    wlogg = np.ascontiguousarray(wlogg, dtype=float)
    wmu = np.ascontiguousarray(wmu, dtype=float)
    jteff = np.ascontiguousarray(jteff, dtype=int)
    jlogg = np.ascontiguousarray(jlogg, dtype=int)
    jmu = np.ascontiguousarray(jmu, dtype=int)
    area = np.ascontiguousarray(area, dtype=float)
    val_mu = np.ascontiguousarray(val_mu, dtype=float)
    wwav = np.ascontiguousarray(wwav, dtype=float)
    jwav = np.ascontiguousarray(jwav, dtype=int)
    nsurf = jteff.size
    nwav = grid.shape[-1]
    fl = np.zeros(nwav, dtype=float)
    if os.uname()[0] == 'Darwin':
        extra_compile_args = extra_link_args = ['-O3']
        headers = ['<cmath>']
    else:
        extra_compile_args = extra_link_args = ['-O3 -fopenmp']
        headers = ['<omp.h>','<cmath>']
    get_flux = weave.inline(code, ['grid', 'wteff', 'wlogg', 'wwav', 'jteff', 'jlogg', 'jwav', 'area', 'val_mu', 'nsurf', 'nwav', 'fl'], type_converters=weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=headers, libraries=['m'], verbose=2)
    tmp = get_flux
    return fl

def Interp_spectroscopy(grid, wteff, wlogg, wmu, jteff, jlogg, jmu, area, val_mu, wav, wav0, dwav):
    """
    Simple interpolation of an atmosphere grid having axes (logtemp, logg, mu, wav).

    This grid interpolation is made for a grid which is regular and linear in
    the wavelength space.

    Note: Because of the Doppler shift, the interpolation on the wavelength
        will necessarily go out of bound, on the lower or upper range. We
        assume that the atmosphere grid has a broader spectral coverage than
        the data.

    Parameters
    ----------
    The interpolation takes a set of points to be interpolated.
    grid : ndarray
        Atmosphere grid, with dimensions (logtemp, logg, mu, wav).
    wteff, wlogg, wmu : ndarray
        Weights of the temperature, logg, mu to interpolate at.
    jteff, jlogg, jmu : ndarray
        Lower index of the position of the temperature, logg, mu to interpolate at.
    area : ndarray
        Area (i.e. weight) of each surface element for the summation.
    val_mu : ndarray
        Value of the cross-section visible to us.
    wav : ndarray
        Values of the wavelength to interpolate at.
    wav0 : float
        Lower wavelength value of the grid.
    dwav : float
        Spacing of the grid.

    Returns
    -------
    spectrum : ndarray
        Spectrum integrated over the surface.
    """
    logger.log(9, "start")
    code = """
    #pragma omp parallel shared(grid,wteff,wlogg,wmu,jteff,jlogg,jmu,area,val_mu,wav,wav0,dwav,nsurf,nwav,nwav_arr,fl) default(none)
    {
    double w1teff, w0teff, w1logg, w0logg, w1mu, w0mu, w1wav, w0wav, tmp_fl, tmp_wav;
    int j0teff, j1teff, j0logg, j1logg, j0mu, j1mu, j0wav, j1wav;
    #pragma omp for
    for (int i=0; i<nsurf; i++) {
        //std::cout << "Surface index: " << i << std::endl;
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
        for (int k=0; k<nwav; k++) {
            //std::cout << "  Wav index: " << k << std::endl;
            //std::cout << "  wav(k): " << wav(k) << ", wav0: " << wav0 << ", dwav: " << dwav << std::endl;
            tmp_wav = (wav(k)-wav0)/dwav;
            //std::cout << "  tmp_wav: " << tmp_wav << std::endl;
            j0wav = (int)tmp_wav;
            w1wav = fmod(tmp_wav,1.);
            w0wav = 1.-w1wav;
            j1wav = j0wav+1;
            if (j0wav < 0){
                j0wav = 0;
                j1wav = 0;
            } else if (j1wav >= nwav_arr){
                j0wav = nwav_arr-1;
                j1wav = nwav_arr-1;
            }
            //std::cout << "  j0teff: " << j0teff << ", j0logg: " << j0logg << ", j0mu: " << j0mu << ", j0wav: " << j0wav << std::endl;
            //std::cout << "  w0teff: " << w0teff << ", w0logg: " << w0logg << ", w0mu: " << w0mu << ", w0wav: " << w0wav << std::endl;
            tmp_fl = \
                ( w1mu * \
                    ( w0wav * \
                        ( w0logg * \
                            ( w0teff * grid(j0teff,j0logg,j1mu,j0wav) + w1teff * grid(j1teff,j0logg,j1mu,j0wav) ) \
                        + w1logg * \
                            ( w0teff * grid(j0teff,j1logg,j1mu,j0wav) + w1teff * grid(j1teff,j1logg,j1mu,j0wav) ) \
                        ) \
                    + w1wav * \
                        ( w0logg * \
                            ( w0teff * grid(j0teff,j0logg,j1mu,j1wav) + w1teff * grid(j1teff,j0logg,j1mu,j1wav) ) \
                        + w1logg * \
                            ( w0teff * grid(j0teff,j1logg,j1mu,j1wav) + w1teff * grid(j1teff,j1logg,j1mu,j1wav) ) \
                        ) \
                    ) \
                + w0mu * \
                    ( w0wav * \
                        ( w0logg * \
                            ( w0teff * grid(j0teff,j0logg,j0mu,j0wav) + w1teff * grid(j1teff,j0logg,j0mu,j0wav) ) \
                        + w1logg * \
                            ( w0teff * grid(j0teff,j1logg,j0mu,j0wav) + w1teff * grid(j1teff,j1logg,j0mu,j0wav) ) \
                        ) \
                    + w1wav * \
                        ( w0logg * \
                            ( w0teff * grid(j0teff,j0logg,j0mu,j1wav) + w1teff * grid(j1teff,j0logg,j0mu,j1wav) ) \
                        + w1logg * \
                            ( w0teff * grid(j0teff,j1logg,j0mu,j1wav) + w1teff * grid(j1teff,j1logg,j0mu,j1wav) ) \
                        ) \
                    ) \
                );
            //std::cout << "tmp_fl " << tmp_fl << std::endl;
            //std::cout << "area*val_mu " << area(i) * val_mu(i) << std::endl;
            //std::cout << "fl " << tmp_fl * area(i) * val_mu(i) << std::endl;
            fl(k) += exp(tmp_fl) * area(i) * val_mu(i);
        }
    }
    }
    """
    #grid, wteff, wlogg, wmu, jteff, jlogg, jmu, area, val_mu, wav, wav0, dwav
    grid = np.ascontiguousarray(grid, dtype=float)
    wteff = np.ascontiguousarray(wteff, dtype=float)
    wlogg = np.ascontiguousarray(wlogg, dtype=float)
    wmu = np.ascontiguousarray(wmu, dtype=float)
    jteff = np.ascontiguousarray(jteff, dtype=int)
    jlogg = np.ascontiguousarray(jlogg, dtype=int)
    jmu = np.ascontiguousarray(jmu, dtype=int)
    area = np.ascontiguousarray(area, dtype=float)
    val_mu = np.ascontiguousarray(val_mu, dtype=float)
    wav = np.ascontiguousarray(wav, dtype=float)
    wav0 = float(wav0)
    dwav = float(dwav)
    nsurf = jteff.size
    nwav = wav.size
    nwav_arr = grid.shape[-1]
    fl = np.zeros(nwav, dtype=float)
    if os.uname()[0] == 'Darwin':
        extra_compile_args = extra_link_args = ['-O3']
        headers = ['<cmath>']
    else:
        extra_compile_args = extra_link_args = ['-O3 -fopenmp']
        headers = ['<omp.h>','<cmath>']
    get_flux = weave.inline(code, ['grid', 'wteff', 'wlogg', 'wmu', 'jteff', 'jlogg', 'jmu', 'area', 'val_mu', 'wav', 'wav0', 'dwav', 'nsurf', 'nwav', 'nwav_arr', 'fl'], type_converters=weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=headers, libraries=['m'], verbose=2)
    tmp = get_flux
    logger.log(9, "end")
    return fl

def Interp_spectroscopy_doppler(grid, wteff, wlogg, wmu, jteff, jlogg, jmu, area, val_mu, wav, wav0, dwav, val_vel):
    """
    Simple interpolation of an atmosphere grid having axes
    (logtemp, logg, mu, wav).

    This grid interpolation is made for a grid which is regular and linear in
    the wavelength space.

    This grid interpolation takes into account the Doppler shift due to each
    surface element.

    This grid interpolation also takes into account the Doppler boosting
    component. I_nu/nu^3 is a Lorentz invariant (and hence I_lambda/nu^5).
    Therefore,
        I(nu) = (nu/nu')^3 I'(nu')
        or
        I(lambda) = (nu/nu')^5 I'(lambda')
    where, in the non-relativistic case (v<<c)
        nu/nu' = 1 + v/c
    and
        (nu/nu')^n ~ 1 + n*v/c
    In this case, we have F_lambda and so the boosting is
        F(lambda) = F(lambda') * (1+5v/c)

    Note: Because of the Doppler shift, the interpolation on the wavelength
        will necessarily go out of bound, on the lower or upper range. We
        assume that the atmosphere grid has a broader spectral coverage than
        the data.

    Parameters
    ----------
    grid : ndarray
        Atmosphere grid, with dimensions (logtemp, logg, mu, wav).
        The flux values are natural logarithm in energy per unit time per unit
        solid angle per unit wavelength.
    wteff, wlogg, wmu : ndarray
        Weights of the temperature, logg, mu to interpolate at.
    jteff, jlogg, jmu : ndarray
        Lower index of the position of the temperature, logg, mu to interpolate
        at.
    area : ndarray
        Area (i.e. weight) of each surface element for the summation.
    val_mu : ndarray
        Value of the cross-section visible to us.
    wav : ndarray
        Values of the wavelength to interpolate at.
    wav0 : float
        Lower wavelength value of the grid.
    dwav : float
        Spacing of the grid.
    val_vel : ndarray
        Velocity value (in v/c unit) of each surface element.

    Returns
    -------
    spectrum : ndarray
        Spectrum integrated over the surface.
    """
    logger.log(9, "start")
    code = """
    #pragma omp parallel shared(grid,wteff,wlogg,wmu,jteff,jlogg,jmu,area,val_mu,wav,wav0,dwav,val_vel,nsurf,nwav,nwav_arr,fl) default(none)
    {
    double w1teff, w0teff, w1logg, w0logg, w1mu, w0mu, w1wav, w0wav, tmp_fl, tmp_wav;
    int j0teff, j1teff, j0logg, j1logg, j0mu, j1mu, j0wav, j1wav;
    #pragma omp for
    for (int i=0; i<nsurf; i++) {
        //std::cout << "Surface index: " << i << std::endl;
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
        for (int k=0; k<nwav; k++) {
            //std::cout << "  Wav index: " << k << std::endl;
            //std::cout << "  wav(k): " << wav(k) << ", wav0: " << wav0 << ", dwav: " << dwav << ", val_vel: " << val_vel(i) << std::endl;
            tmp_wav = (wav(k)*(1+val_vel(i))-wav0)/dwav;
            //std::cout << "  tmp_wav: " << tmp_wav << std::endl;
            j0wav = (int)tmp_wav;
            w1wav = fmod(tmp_wav,1.);
            w0wav = 1.-w1wav;
            j1wav = j0wav+1;
            if (j0wav < 0){
                j0wav = 0;
                j1wav = 0;
            } else if (j1wav >= nwav_arr){
                j0wav = nwav_arr-1;
                j1wav = nwav_arr-1;
            }
            //std::cout << "  j0teff: " << j0teff << ", j0logg: " << j0logg << ", j0mu: " << j0mu << ", j0wav: " << j0wav << std::endl;
            //std::cout << "  w0teff: " << w0teff << ", w0logg: " << w0logg << ", w0mu: " << w0mu << ", w0wav: " << w0wav << std::endl;
            tmp_fl = \
                ( w1mu * \
                    ( w0wav * \
                        ( w0logg * \
                            ( w0teff * grid(j0teff,j0logg,j1mu,j0wav) + w1teff * grid(j1teff,j0logg,j1mu,j0wav) ) \
                        + w1logg * \
                            ( w0teff * grid(j0teff,j1logg,j1mu,j0wav) + w1teff * grid(j1teff,j1logg,j1mu,j0wav) ) \
                        ) \
                    + w1wav * \
                        ( w0logg * \
                            ( w0teff * grid(j0teff,j0logg,j1mu,j1wav) + w1teff * grid(j1teff,j0logg,j1mu,j1wav) ) \
                        + w1logg * \
                            ( w0teff * grid(j0teff,j1logg,j1mu,j1wav) + w1teff * grid(j1teff,j1logg,j1mu,j1wav) ) \
                        ) \
                    ) \
                + w0mu * \
                    ( w0wav * \
                        ( w0logg * \
                            ( w0teff * grid(j0teff,j0logg,j0mu,j0wav) + w1teff * grid(j1teff,j0logg,j0mu,j0wav) ) \
                        + w1logg * \
                            ( w0teff * grid(j0teff,j1logg,j0mu,j0wav) + w1teff * grid(j1teff,j1logg,j0mu,j0wav) ) \
                        ) \
                    + w1wav * \
                        ( w0logg * \
                            ( w0teff * grid(j0teff,j0logg,j0mu,j1wav) + w1teff * grid(j1teff,j0logg,j0mu,j1wav) ) \
                        + w1logg * \
                            ( w0teff * grid(j0teff,j1logg,j0mu,j1wav) + w1teff * grid(j1teff,j1logg,j0mu,j1wav) ) \
                        ) \
                    ) \
                );
            //std::cout << "tmp_fl " << tmp_fl << std::endl;
            //std::cout << "area*val_mu " << area(i) * val_mu(i) << std::endl;
            //std::cout << "fl " << tmp_fl * area(i) * val_mu(i) << std::endl;
            fl(k) += exp(tmp_fl) * area(i) * val_mu(i) * (1+5*val_vel(i));
        }
    }
    }
    """
    #grid, wteff, wlogg, wmu, jteff, jlogg, jmu, area, val_mu, wav, wav0, dwav
    grid = np.ascontiguousarray(grid, dtype=float)
    wteff = np.ascontiguousarray(wteff, dtype=float)
    wlogg = np.ascontiguousarray(wlogg, dtype=float)
    wmu = np.ascontiguousarray(wmu, dtype=float)
    jteff = np.ascontiguousarray(jteff, dtype=int)
    jlogg = np.ascontiguousarray(jlogg, dtype=int)
    jmu = np.ascontiguousarray(jmu, dtype=int)
    area = np.ascontiguousarray(area, dtype=float)
    val_mu = np.ascontiguousarray(val_mu, dtype=float)
    wav = np.ascontiguousarray(wav, dtype=float)
    wav0 = float(wav0)
    dwav = float(dwav)
    val_vel = np.ascontiguousarray(val_vel, dtype=float)
    nsurf = jteff.size
    nwav = wav.size
    nwav_arr = grid.shape[-1]
    fl = np.zeros(nwav, dtype=float)
    if os.uname()[0] == 'Darwin':
        extra_compile_args = extra_link_args = ['-O3']
        headers = ['<cmath>']
    else:
        extra_compile_args = extra_link_args = ['-O3 -fopenmp']
        headers = ['<omp.h>','<cmath>']
    get_flux = weave.inline(code, ['grid', 'wteff', 'wlogg', 'wmu', 'jteff', 'jlogg', 'jmu', 'area', 'val_mu', 'wav', 'wav0', 'dwav', 'val_vel', 'nsurf', 'nwav', 'nwav_arr', 'fl'], type_converters=weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=headers, libraries=['m'], verbose=2)
    tmp = get_flux
    logger.log(9, "end")
    return fl

