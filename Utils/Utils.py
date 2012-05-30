# Licensed under a 3-clause BSD style license - see LICENSE

from .import_modules import *


def Err_velocity(chi2, vels, dof, clip=None, normalize=False, redchi2_unity=False, verbose=False):
    """Err_velocity(chi2, vels, dof, clip=None, normalize=False, redchi2_unity=False, verbose=False)
    Given a vector of chi2 values and associated velocity shifts,
    will return the best chi2, the best velocity shift and the
    1sigma error bar by approximating the region near the minimum
    chi2 as a parabola.
    
    chi2: vector of chi2 values.
    vels: vector of velocities.
    dof: number of degrees of freedom.
    clip (None): range, in m/s, around the minimum to fit the parabola.
    normalize (False): Whether the chi2 should be divided by 'dof'.
    redchi2_unity (False): Normalize the error so that reduced chi2 is unity.
    verbose (False): Plot the chi2 vector and fitted parabola.

    >>> chi2fit, vfit, err_vfit = Err_velocity(chi2, vels, dof)
    """
    if normalize:
        chi2 = chi2/dof
    if clip is not None:
    #thres = chi2.min()*(1.+2.*clip**2/dof)
        inds = abs(vels-vels[chi2.argmin()]) <= clip
        y = chi2[inds]
        x = vels[inds]
    else:
        inds = None
        y = chi2
        x = vels
    a_n = GPolynomial_fit(y, x, coeff=3)
    p_n = numpy.poly1d(a_n)
    vfit = -0.5*a_n[1]/a_n[0]
    chi2fit = p_n(vfit)
    err_vfit = numpy.sqrt(1/a_n[0])
    if redchi2_unity:
        err_vfit *= numpy.sqrt(chi2fit/dof)
    if verbose:
        plotxy(chi2, vels, symbol=3)
        plotxy(y, x, line=None, symbol=3, color=4)
        plotxy(p_n(vels), vels, color=2)
    return chi2fit, vfit, err_vfit

def Extinction(w, Rv=3.1, cardelli=False):
    """Extinction(w, Rv=3.1)
    Returns the selective extinction for A_V=1 for a set of wavelengths w.
    
    w: wavelengths in microns.
    Rv (3.1): ratio of total to selective extinction (A_V/E(B-V)).
        standard value is Rv=3.1.
    cardelli (False): If true will use Cardelli et al. 1989.
        If false, will use O'Donnell 1994.
    
    
    Source:
    Cardelli, Clayton and Mathis, 1989, ApJ, 345, 245:
    http://adsabs.harvard.edu/abs/1989ApJ...345..245C
    
    O'Donnell, J. E., 1994, ApJ, 422, 158:
    http://adsabs.harvard.edu/abs/1994ApJ...422..158O
    Note: 1/w values between 1.1 micron^-1 < x < 3.3 microns^-1 (near-IR, optical)
    
    """
    x = numpy.atleast_1d(1/w)
    ext = numpy.zeros_like(x)
    #
    inds = x < 0.3
    ext[inds] = numpy.nan
    #
    inds = (x >= 0.3) * (x < 1.1)
    if inds.any():
        y = x[inds]
        p_a = 0.574 * y**1.61
        p_b = -0.527 * y**1.61
        ext[inds] = p_a + p_b/Rv
    #
    inds = (x >= 1.1) * (x < 3.3)
    if inds.any():
        y = x[inds]-1.82
        ##### Using Cardelli, Clayton and Mathis (1989)
        if cardelli:
            p_a = numpy.poly1d([0.32999, -0.77530, 0.01979, 0.72085, -0.02427, -0.50447, 0.17699, 1.])(y)
            p_b = numpy.poly1d([-2.09002, 5.30260, -0.62251, -5.38434, 1.07233, 2.28305, 1.41338, 0.])(y)
        ##### Using O'Donnell (1994)
        else:
            p_a = numpy.poly1d([-0.505, 1.647, -0.827, -1.718, 1.137, 0.701, -0.609, 0.104, 1.])(y)
            p_b = numpy.poly1d([3.347, -10.805, 5.491, 11.102, -7.985, -3.989, 2.908, 1.952, 0.])(y)
        ext[inds] = p_a + p_b/Rv
    #
    inds = (x >= 3.3) * (x < 5.9)
    if inds.any():
        y = x[inds]
        p_a = 1.752 - 0.316*y - 0.104/((y-4.67)**2 + 0.341)
        p_b = -3.090 + 1.825*y + 1.206/((y-4.62)**2 + 0.263)
        ext[inds] = p_a + p_b/Rv
    #
    inds = (x >= 5.9) * (x < 8.0)
    if inds.any():
        y = x[inds]
        f_a = -0.04473*(y-5.9)**2 - 0.009779*(y-5.9)**3
        f_b = 0.2130*(y-5.9)**2 + 0.1207*(y-5.9)**3
        p_a = 1.752 - 0.316*y - 0.104/((y-4.67)**2 + 0.341)
        p_b = -3.090 + 1.825*y + 1.206/((y-4.62)**2 + 0.263)
        ext[inds] = (p_a + f_a) + (p_b + f_b)/Rv
    #
    inds = (x >= 8.0) * (x <= 10.0)
    if inds.any():
        y = x[inds]
        p_a = -1.073 - 0.628*(y-8) + 0.137*(y-8)**2 - 0.070*(y-8)**3
        p_b = 13.670 + 4.257*(y-8) - 0.420*(y-8)**2 + 0.374*(y-8)**3
        ext[inds] = p_a + p_b/Rv
    #
    inds = x > 10.0
    ext[inds] = numpy.nan
    
    ##### Return scalar if possible
    if ext.shape == (1,):
        return ext[0]
    return ext

def fit_linear(y, x=None, err=1.0, m=None, b=None, output=None, inline=False):
    """
    fit_linear(y, x=None, err=1.0, m=None, b=None, output=None, inline=False):
    return (sol, res, rank, s)
    Uses the scipy.linalg.lstsq function to solve the equation y = mx + b
    sol -> [b, m]
    N.B. Uses the scipy.linalg.lstsq algorithm.
    If inline = True, flattens the results.
    """
    #x = array([52997., 53210., 53310., 53380.])
    #y = array([1.66, 1.54, 1.4, 1.4])
    # standard error of the y-variable:
    #sy = array([0.05, 0.05, 0.05, 0.05])

    if x is None:
        x = numpy.arange(y.shape[0], dtype=float)
    if (b is not None) and (m is not None):
        sol = [b, m]
        res = (((b + m*x - y)/err)**2).sum()
        rank = 0.
        s = 0.
    else:
        if b is not None:
            A = numpy.reshape(x/err,(x.shape[0],1))
            y1 = y-b
            y1 /= err
            sol, res, rank, s = scipy.linalg.lstsq(A, y1)
            sol = [b,sol[0]]
        elif m is not None:
            A = numpy.resize(1/err,(x.shape[0],1))
            y1 = y-m*x
            y1 /= err
            sol, res, rank, s = scipy.linalg.lstsq(A, y1)
            sol = [sol[0],m]
        else:
            A = (numpy.vstack([numpy.ones(x.shape[0], dtype=float),x])/err).T
            y1 = y/err
            sol, res, rank, s = scipy.linalg.lstsq(A, y1)
    if output:
        b, m = sol
        fit_y = b + m*x
        print 'b -> ' + str(b)
        print 'm -> ' + str(m)
        print 'Reduced chi-square: ' + str(res/(len(y)-rank))
        plotxy(y, x, line=None, symbol=2, color=2)
        plotxy(fit_y, x)
    if res.shape == (0,):
        res = numpy.r_[0.]
    if inline:
        return numpy.hstack((sol, res, rank, s))
    else:
        return (sol, res, rank, s)

def Get_potential(x, y, z, q, omega=1.):
    qp1by2om2 = (q+1.)/2.*omega**2
    rc2 = x**2+y**2+z**2
    rx = numpy.sqrt(rc2+1-2*x)
    rc = numpy.sqrt(rc2)
    psi = 1/rc + q/rx - q*x + qp1by2om2*(rc2-z**2)
    dpsi = -1/rc**3-q/rx**3
    dpsidx = x*(dpsi+2*qp1by2om2)+q*(-1+1/rx**3)
    dpsidy = y*(dpsi+2*qp1by2om2)
    dpsidz = z*dpsi
    return rc, rx, dpsi, dpsidx, dpsidy, dpsidz, psi

def Get_radius(r, cosx, cosy, cosz, psi0=72.2331279572, q=63.793154816, omega=1.):
    x = r*cosx
    y = r*cosy
    z = r*cosz
    rc, rx, dpsi, dpsidx, dpsidy, dpsidz, psi = Get_potential(x, y, z, q, omega)
    dpsidr = dpsidx*cosx+dpsidy*cosy+dpsidz*cosz
    dr = -(psi-psi0)/dpsidr
    return dr

def Get_saddle(x, q, omega=1.):
    qp1by2om2 = (q+1.)/2.*omega**2
    rc, rx, dpsi, dpsidx, dpsidy, dpsidz, psi = Get_potential(x, 0., 0., q, omega)
    d2psidx2 = dpsi+3.*(x**2/rc**5+q*(x-1.)**2/rx**5)+2.*qp1by2om2
    dx = -dpsidx/d2psidx2
    x = x+dx
    return dx/x

def Get_K_to_q(porb, xsini):
    """Get_K_to_q(porb, xsini)
    Returns the K_to_q conversion factor given an
    orbital period and an observed xsini.
    The K is for the "primary" whereas xsini is that
    of the "secondary" star.
    
    porb: Orbital period in seconds.
    xsini: Projected semi-major axis in light-second.
    
    >>> K_to_q = Get_K_to_q(porb, xsini)
    """
    return porb / (TWOPI * xsini * cts.c)

def Limb_darkening(lam, mu):
    """
    Returns the limb darkening.
    
    lam: wavelength in micrometer.
        nlam = lam.shape
    mu: cos(theta) direction of emission angle.
        nmu, 1 = mu.shape
    
    Note: Only valid for 0.42257 < lam < 1.100 micrometer.
    From Neckel 2005.
    """
    def L_422_1100(lam,mu):
        a_00 = 0.75267
        a_01 = -0.265577
        a_10 = 0.93874
        a_11 = 0.265577
        a_15 = -0.004095
        a_20 = -1.89287
        a_25 = 0.012582
        a_30 = 2.42234
        a_35 = -0.017117
        a_40 = -1.71150
        a_45 = 0.011977
        a_50 = 0.49062
        a_55 = -0.003347
        lam5 = lam**5
        a_0 = a_00 + a_01/lam
        a_1 = a_10 + a_11/lam + a_15/lam5
        a_2 = a_20 + a_25/lam5
        a_3 = a_30 + a_35/lam5
        a_4 = a_40 + a_45/lam5
        a_5 = a_50 + a_55/lam5
        return a_0 + (a_1 + (a_2 + (a_3 + (a_4 + a_5*mu )*mu )*mu )*mu )*mu
    def L_385_422(lam,mu):
        a_00 = 0.09900
        a_01 = 0.010643
        a_10 = 1.96884
        a_11 = -0.010643
        a_15 = -0.009166
        a_20 = -2.80511
        a_25 = 0.024873
        a_30 = 3.32780
        a_35 = -0.029317
        a_40 = -2.17878
        a_45 = 0.018273
        a_50 = 0.58825
        a_55 = -0.004663
        lam5 = lam**5
        a_0 = a_00 + a_01/lam
        a_1 = a_10 + a_11/lam + a_15/lam5
        a_2 = a_20 + a_25/lam5
        a_3 = a_30 + a_35/lam5
        a_4 = a_40 + a_45/lam5
        a_5 = a_50 + a_55/lam5
        return a_0 + (a_1 + (a_2 + (a_3 + (a_4 + a_5*mu )*mu )*mu )*mu )*mu
    def L_300_372(lam,mu):
        a_00 = 0.35601
        a_01 = -0.085217
        a_10 = 1.11529
        a_11 = 0.085217
        a_15 = -0.001871
        a_20 = -0.67237
        a_25 = 0.003589
        a_30 = 0.18696
        a_35 = -0.002415
        a_40 = 0.00038
        a_45 = 0.000897
        a_50 = 0.01373
        a_55 = -0.000200
        lam5 = lam**5
        a_0 = a_00 + a_01/lam
        a_1 = a_10 + a_11/lam + a_15/lam5
        a_2 = a_20 + a_25/lam5
        a_3 = a_30 + a_35/lam5
        a_4 = a_40 + a_45/lam5
        a_5 = a_50 + a_55/lam5
        return a_0 + (a_1 + (a_2 + (a_3 + (a_4 + a_5*mu )*mu )*mu )*mu )*mu
    limb = numpy.empty((mu.shape[0],lam.shape[0]))
    inds = lam<0.37298
    if inds.any():
        limb[:,inds] = L_300_372(lam[inds],mu)
    inds = (lam<0.42257)*(lam>0.37298)
    if inds.any():
        limb[:,inds] = L_385_422(lam[inds],mu)
    inds = lam>0.42257
    if inds.any():
        limb[:,inds] = L_422_1100(lam[inds],mu)
    return limb

def Mass_companion(mass_function, q, incl):
    """
    Returns the mass of the neutron star companion.
    
    mass_function: Mass function of the neutron star.
    q: Mass ratio (q = M_ns/M_wd)
    incl: Orbital inclination in radians.
    
    >>> Mass_companion(mass_function, q, incl)
    """
    return mass_function * (1+q)**2 / numpy.sin(incl)**3

def Mass_ratio(mass_function, M_ns, incl):
    """Mass_ratio(mass_function, M_ns, incl)
    Returns the mass ratio of a binary (q = M_ns/M_wd).
    
    mass_function: Mass function of the binary.
    M_ns: Neutron star mass in solar mass.
    incl: Orbital inclination in radians.
    
    >>> Mass_ratio(mass_function, M_ns, incl)
    """
    q = -1./9
    r = 0.5*M_ns*numpy.sin(incl)**3/mass_function + 1./27
    s = (r + numpy.sqrt(q**3+r**2))**(1./3)
    t = (r - numpy.sqrt(q**3+r**2))**(1./3)
    return s+t-2./3

def Orbital_separation(asini, q, incl):
    """Orbital_separation(asini, q, incl)
    Returns the orbital separation in centimeters.
    
    asini: Measured a_ns*sin(incl), in light-seconds.
    q: Mass ratio of the binary (q = M_ns/M_wd).
    incl: Orbital inclination in radians.
    
    >>> Orbital_separation(asini, q, incl)
    """
    return asini*(1+q)/numpy.sin(incl)*C*100

def Potential(x, y, z, q, qp1by2om2):
    """
    Returns the potential at a given point (x, y, z) given
    a mass ratio and a qp1by2om2.
    
    x,y,z can be vectors or scalars.
    q: mass ratio (mass companion/mass pulsar).
    qp1by2om2: (q+1) / (2 * omega^2)
    
    >>> rc, rx, dpsi, dpsidx, dpsidy, dpsidz, psi = Potential(x, y, z, q, qp1by2om2)
    """
    rc2 = x**2+y**2+z**2
    rx = numpy.sqrt(rc2+1-2*x)
    rc = numpy.sqrt(rc2)
    psi = 1/rc + q/rx - q*x + qp1by2om2*(rc2-z**2)
    dpsi = -1/rc**3-q/rx**3
    dpsidx = x*(dpsi+2*qp1by2om2)+q*(-1+1/rx**3)
    dpsidy = y*(dpsi+2*qp1by2om2)
    dpsidz = z*dpsi
    return rc, rx, dpsi, dpsidx, dpsidy, dpsidz, psi

def Resample_linlog(xold):
    """Resample_linlog(xold)
    Resample a linear wavelength vector to a log space and
    returns the new vector, the v/c value and the redshift z.
    
    >>> xnew, v, z = Resample_linlog(xold)
    """
    z = (xold[-1]-xold[-2])/xold[-2]
    v = scipy.optimize.fsolve( lambda beta: (1+z)**2-(1+beta)/(1-beta), 0.)
    xnew = xold[0] * (1+z)**numpy.arange(numpy.ceil(numpy.log(xold[-1]/xold[0])/numpy.log(xold[-1]/xold[-2])) + 1)
    #v = 1 - xold[-1]/xold[-2]
    #xnew = xold[0] * (1-v)**numpy.arange(numpy.ceil(numpy.log(xold[-1]/xold[0])/numpy.log(xold[-1]/xold[-2])) + 1)
    return xnew, numpy.abs(v), z



######################## code using scipy.weave ########################

def Getaxispos_scalar(xx, x):
    """ Getaxispos_scalar(xx, x)
    Given a scalar x, returns the index and fractional weight
    that corresponds to the nearest linear interpolation from
    the vector xx.
    
    xx: vector of values to be interpolated from.
    x: scalar value to be interpolated.
    
    weight,index = Getaxispos_scalar(xx, x)
    """
    n = xx.shape[0]
    code = """
    int jl, ju, jm;
    double w;
    bool ascending = xx(n-1) > xx(0);
    jl = 0;
    ju = n;
    while ((ju-jl) > 1)
    {
        jm = (ju+jl)/2;
        if (ascending == (x > xx(jm)))
            jl = jm;
        else
            ju = jm;
    }
    jl = (jl < (n-1) ? jl : n-2);
    w = (x-xx(jl))/(xx(jl+1)-xx(jl));
    py::tuple results(2);
    results[0] = w;
    results[1] = jl;
    return_val = results;
    """
    xx = numpy.asarray(xx)
    x = float(x)
    get_axispos = scipy.weave.inline(code, ['xx', 'x', 'n'], type_converters=scipy.weave.converters.blitz, compiler='gcc', verbose=2)
    w,j = get_axispos
    return w,j

def Getaxispos_vector(xx, x):
    """ Getaxispos_vector(xx, x)
    Given a vector x, returns the indices and fractional weights
    that corresponds to their nearest linear interpolation from
    the vector xx.
    
    xx: vector of values to be interpolated from.
    x: vector of values to be interpolated.
    
    weights,indices = Getaxispos_scalar(xx, x)
    """
    n = xx.shape[0]
    m = x.shape[0]
    #xx = numpy.asarray(xx, dtype=float)
    #x = numpy.asarray(x, dtype=float)
    j = numpy.empty(m, dtype=int)
    w = numpy.empty(m, dtype=float)
    code = """
    #pragma omp parallel shared(xx,x,n,m,j,w) default(none)
    {
    int jl, ju, jm;
    double a;
    bool ascending = xx(n-1) > xx(0);
    #pragma omp for
    for (int i=0; i<m; ++i) {
       //std::cout << i << " " << x(i) << std::endl;
       jl = 0;
       ju = n;
       while ((ju-jl) > 1)
       {
           //std::cout << "+++" << std::endl;
           //std::cout << jl << " " << ju << " " << jm << std::endl;
           jm = (ju+jl)/2;
           //std::cout << i << " " << x(i) << " " << xx(jm) << std::endl;
           if (ascending == (x(i) > xx(jm)))
               jl = jm;
           else
               ju = jm;
           //std::cout << jl << " " << ju << " " << jm << std::endl;
       }
       j(i) = (jl < (n-1) ? jl : n-2);
       w(i) = (x(i)-xx(j(i)))/(xx(j(i)+1)-xx(j(i)));
    }
    }
    """
    try:
        if os.uname()[0] == 'Darwin':
            extra_compile_args = extra_link_args = ['-O3']
        else:
            extra_compile_args = extra_link_args = ['-O3 -fopenmp']
        get_axispos = scipy.weave.inline(code, ['xx', 'x', 'n', 'm', 'j', 'w'], type_converters=scipy.weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=['<omp.h>'], verbose=2)
    except:
        get_axispos = scipy.weave.inline(code, ['xx', 'x', 'n', 'm', 'j', 'w'], type_converters=scipy.weave.converters.blitz, compiler='gcc', extra_compile_args=['-O3'], extra_link_args=['-O3'], verbose=2)
    tmp = get_axispos
    return w,j

def Inter8_photometry(grid, wteff, wlogg, wmu, jteff, jlogg, jmu, area, val_mu):
    """
    """
    code = """
    double fl = 0.;
    #pragma omp parallel shared(grid,wteff,wlogg,wmu,jteff,jlogg,jmu,area,val_mu,nsurf,fl) default(none)
    {
    double w1teff, w0teff, w1logg, w0logg, w1mu, w0mu, tmp_fl;
    int j0teff, j1teff, j0logg, j1logg, j0mu, j1mu, nlow, nhigh;
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
    int j0teff, j1teff, j0logg, j1logg, j0mu, j1mu, nlow, nhigh;
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

def Inter8_doppler(grid, wteff, wlogg, wmu, wlam, jteff, jlogg, jmu, jlam, area, val_mu):
    """
    Note: Because of the Doppler shift, the interpolation on the wavelength
        will necessarily go out of bound, on the lower or upper range. We 
        assume that the atmosphere grid has a broader spectral coverage than
        the data.
        
        ### Check the fl(k) += foo
    """
    code = """
    #pragma omp parallel shared(grid,wteff,wlogg,wmu,wlam,jteff,jlogg,jmu,jlam,area,val_mu,nsurf,nlam,fl) default(none)
    {
    double w1teff, w0teff, w1logg, w0logg, w1mu, w0mu, w1lam, w0lam;
    int j0teff, j1teff, j0logg, j1logg, j0mu, j1mu, j0lam, j1lam, nlow, nhigh;
    #pragma omp for
    for (int k=0; k<nlam; k++) {
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
            w1lam = wlam(i);
            w0lam = 1.-w1lam;
            j0lam = jlam(i);
            j1lam = 1.+j0lam;
            fl(k) += (w1mu*(w0lam*(w0logg*(w0teff*grid(j0teff,j0logg,j1mu,k+j0lam) + w1teff*grid(j1teff,j0logg,j1mu,k+j0lam)) \
                        + w1logg*(w0teff*grid(j0teff,j1logg,j1mu,k+j0lam) + w1teff*grid(j1teff,j1logg,j1mu,k+j0lam))) \
                    + w1lam*(w0logg*(w0teff*grid(j0teff,j0logg,j1mu,k+j1lam) + w1teff*grid(j1teff,j0logg,j1mu,k+j1lam)) \
                        + w1logg*(w0teff*grid(j0teff,j1logg,j1mu,k+j1lam) + w1teff*grid(j1teff,j1logg,j1mu,k+j1lam)))) \
                    + w0mu*(w0lam*(w0logg*(w0teff*grid(j0teff,j0logg,j0mu,k+j0lam) + w1teff*grid(j1teff,j0logg,j0mu,k+j0lam)) \
                        + w1logg*(w0teff*grid(j0teff,j1logg,j0mu,k+j0lam) + w1teff*grid(j1teff,j1logg,j0mu,k+j0lam))) \
                    + w1lam*(w0logg*(w0teff*grid(j0teff,j0logg,j0mu,k+j1lam) + w1teff*grid(j1teff,j0logg,j0mu,k+j1lam)) \
                        + w1logg*(w0teff*grid(j0teff,j1logg,j0mu,k+j1lam) + w1teff*grid(j1teff,j1logg,j0mu,k+j1lam))))) * area(i) * val_mu(i);
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
    get_flux = scipy.weave.inline(code, ['grid', 'wteff', 'wlogg', 'wmu', 'wlam', 'jteff', 'jlogg', 'jmu', 'jlam', 'area', 'val_mu', 'nsurf', 'nlam', 'fl'], type_converters=scipy.weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=['<omp.h>','<cmath>'], libraries=['m'], verbose=2)
    tmp = get_flux
    return fl

def Inter8_doppler_savememory_old(grid, wteff, wlogg, wmu, wlam, jteff, jlogg, jmu, jlam, mu_grid):
    """
    """
    code = """
    #pragma omp parallel shared(grid,wteff,wlogg,wmu,wlam,jteff,jlogg,jmu,jlam,mu_grid,nsurf,nlam,fl) default(none)
    {
    double w1teff, w0teff, w1logg, w0logg, w1mu, w0mu, w1lam, w0lam;
    int j0teff, j1teff, j0logg, j1logg, j0mu, j1mu, j0lam, j1lam, nlow, nhigh;
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
        w1lam = wlam(i);
        w0lam = 1.-w1lam;
        j0lam = jlam(i);
        j1lam = 1.+j0lam;
        if (j0lam<0) {
            nlow = -j0lam;
            nhigh = 0;
            for (int k=0; k<nlow; k++){
                fl(i,k) = (w1mu*mu_grid(j1mu,0) + w0mu*mu_grid(j0mu,0)) * \
                            (w0logg*(w0teff*grid(j0teff,j0logg,0) + w1teff*grid(j1teff,j0logg,0)) \
                           + w1logg*(w0teff*grid(j0teff,j1logg,0) + w1teff*grid(j1teff,j1logg,0)));
            }
        } else {
            nlow = 0;
            nhigh = j0lam;
            for (int k=0; k<nhigh; k++){
                fl(i,nlam-k-1) = (w1mu*mu_grid(j1mu,nlam-1) + w0mu*mu_grid(j0mu,nlam-1)) * \
                                    (w0logg*(w0teff*grid(j0teff,j0logg,nlam-1) + w1teff*grid(j1teff,j0logg,nlam-1)) \
                                   + w1logg*(w0teff*grid(j0teff,j1logg,nlam-1) + w1teff*grid(j1teff,j1logg,nlam-1)));
            }
        }
        for (int k=0+nlow; k<nlam-nhigh; k++) {
            fl(i,k) = (w1mu*mu_grid(j1mu,k+j0lam) + w0mu*mu_grid(j0mu,k+j0lam)) * \
                        (w0lam*(w0logg*(w0teff*grid(j0teff,j0logg,k+j0lam) + w1teff*grid(j1teff,j0logg,k+j0lam)) + \
                         w1logg*(w0teff*grid(j0teff,j1logg,k+j0lam) + w1teff*grid(j1teff,j1logg,k+j0lam)))) + \
                      (w1mu*mu_grid(j1mu,k+j1lam) + w0mu*mu_grid(j0mu,k+j1lam)) * \
                        (w1lam*(w0logg*(w0teff*grid(j0teff,j0logg,k+j1lam) + w1teff*grid(j1teff,j0logg,k+j1lam)) + \
                         w1logg*(w0teff*grid(j0teff,j1logg,k+j1lam) + w1teff*grid(j1teff,j1logg,k+j1lam))));
            }
    }
    }
    """
    nsurf = jteff.size
    nlam = grid.shape[-1]
    fl = numpy.empty((nsurf,nlam), dtype=float)
    if os.uname()[0] == 'Darwin':
        extra_compile_args = extra_link_args = ['-O3']
    else:
        extra_compile_args = extra_link_args = ['-O3 -fopenmp']
    get_flux = scipy.weave.inline(code, ['grid', 'wteff', 'wlogg', 'wmu', 'wlam', 'jteff', 'jlogg', 'jmu', 'jlam', 'mu_grid', 'nsurf', 'nlam', 'fl'], type_converters=scipy.weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=['<omp.h>','<cmath>'], libraries=['m'], verbose=2)
    tmp = get_flux
    return fl

def Inter8_doppler_savememory(grid, wteff, wlogg, wmu, wlam, jteff, jlogg, jmu, jlam, mu_grid, area, val_mu):
    """
    This grid interpolation is made for a grid which is linear in the velocity
    or redshift space, e.g. log lambda,
    Note: Because of the Doppler shift, the interpolation on the wavelength
        will necessarily go out of bound, on the lower or upper range. We 
        assume that the atmosphere grid has a broader spectral coverage than
        the data.
    """
    code = """
    #pragma omp parallel shared(grid,wteff,wlogg,wmu,wlam,jteff,jlogg,jmu,jlam,mu_grid,area,val_mu,nsurf,nlam,fl) default(none)
    {
    double w1teff, w0teff, w1logg, w0logg, w1mu, w0mu, w1lam, w0lam;
    int j0teff, j1teff, j0logg, j1logg, j0mu, j1mu, j0lam, j1lam, nlow, nhigh;
    #pragma omp for
    for (int k=0; k<nlam; k++) {
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
            w1lam = wlam(i);
            w0lam = 1.-w1lam;
            j0lam = jlam(i);
            j1lam = 1.+j0lam;
            fl(k) += ( \
                (w1mu*mu_grid(j1mu,k+j0lam) + w0mu*mu_grid(j0mu,k+j0lam)) * (w0lam*\
                    (w0logg*(w0teff*grid(j0teff,j0logg,k+j0lam) + w1teff*grid(j1teff,j0logg,k+j0lam)) + \
                    w1logg*(w0teff*grid(j0teff,j1logg,k+j0lam) + w1teff*grid(j1teff,j1logg,k+j0lam)))) + \
                (w1mu*mu_grid(j1mu,k+j1lam) + w0mu*mu_grid(j0mu,k+j1lam)) * (w1lam*\
                    (w0logg*(w0teff*grid(j0teff,j0logg,k+j1lam) + w1teff*grid(j1teff,j0logg,k+j1lam)) + \
                    w1logg*(w0teff*grid(j0teff,j1logg,k+j1lam) + w1teff*grid(j1teff,j1logg,k+j1lam)))) \
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
    get_flux = scipy.weave.inline(code, ['grid', 'wteff', 'wlogg', 'wmu', 'wlam', 'jteff', 'jlogg', 'jmu', 'jlam', 'mu_grid', 'area', 'val_mu', 'nsurf', 'nlam', 'fl'], type_converters=scipy.weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=['<omp.h>','<cmath>'], libraries=['m'], verbose=2)
    tmp = get_flux
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
    int j0teff, j1teff, j0logg, j1logg, j0mu, j1mu, nlow, nhigh;
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
    int j0teff, j1teff, j0logg, j1logg, j0lam, j1lam, nlow, nhigh;
    #pragma omp for
    for (int k=0; k<nlam; k++) {
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
            fl(k) += (w0lam*(w0logg*(w0teff*grid(j0teff,j0logg,k+j0lam) + w1teff*grid(j1teff,j0logg,k+j0lam)) \
                        + w1logg*(w0teff*grid(j0teff,j1logg,k+j0lam) + w1teff*grid(j1teff,j1logg,k+j0lam))) \
                    + w1lam*(w0logg*(w0teff*grid(j0teff,j0logg,k+j1lam) + w1teff*grid(j1teff,j0logg,k+j1lam)) \
                        + w1logg*(w0teff*grid(j0teff,j1logg,k+j1lam) + w1teff*grid(j1teff,j1logg,k+j1lam)))) * area(i) * val_mu(i);
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

def GPolynomial_fit(y, x=None, err=None, coeff=1, Xfnct=None, Xfnct_offset=False):
    """
    Best-fit generalized polynomial to a function minimizing:
    chi2 = sum_i( [y(x_i) - sum_k( a_k * X_k(x_i) )]**2 / err_i**2 )
    X_k(x_i) = O_k(x_i)
        if Xfnct=None, i.e. O_k is a simple polynomial of order k
    X_k(x_i) = O_k(x_i)*f(x_i)
        if Xfnct=f(x_i) and Xfnct_offset=False
    X_k(x_i) = O_k(x_i)*f(x_i) + offset
        if Xfnct=f(x_i) and Xfnct_offset=True

    y: the y values, shape (n)
    x (None): the x values, shape (n)
    err (None): the error values, shape (1) or (n)
    coeff (1): the number of coefficients to the generalized polynomial
            function to be fitted (>= 1)
    Xfnct (None): a function to generalize the polynomial, shape (n)
    Xfnct_offset (False): whether the polynomial includes a constant offset or not

    Returns generalized polynomial coefficients
        shape (coeff)
        i.e. (a_n, a_(n-1), ..., a_1, a_0)
    """
    n = y.size
    if x is None:
        x = numpy.arange(n, dtype=float)
    if err is None:
        err = numpy.ones(n, dtype=float)
    if Xfnct is None:
        Xfnct = numpy.ones(n, dtype=float)
    if Xfnct_offset:
        Xfnct_offset = 1
    else:
        Xfnct_offset = 0
    a = numpy.empty((n,coeff), dtype=float)
    b = numpy.empty(n, dtype=float)
    code = """
    if (Xfnct_offset == 1) {
        for (int i=0; i<n; ++i) {
            for (int k=0; k<coeff; ++k) {
            if (k==0)
                a(i,coeff-1-k) = 1/err(i);
            else if (k==1)
                a(i,coeff-1-k) = Xfnct(i)/err(i);
            else
                a(i,coeff-1-k) = a(i,coeff-k)*x(i);
            }
            b(i) = y(i)/err(i);
        }
    }
    else {
        for (int i=0; i<n; ++i) {
            for (int k=0; k<coeff; ++k) {
            if (k==0)
                a(i,coeff-1-k) = Xfnct(i)/err(i);
            else
                a(i,coeff-1-k) = a(i,coeff-k)*x(i);
            }
            //std::cout << y(i) << " " << err(i) << std::endl;
            b(i) = y(i)/err(i);
        }
    }
    """
    prep_lstsq = scipy.weave.inline(code, ['y', 'x', 'err', 'Xfnct', 'Xfnct_offset', 'a', 'b', 'n', 'coeff'], type_converters=scipy.weave.converters.blitz, compiler='gcc')
    tmp = prep_lstsq
    return numpy.linalg.lstsq(a, b)[0]

def Make_geodesic(n):
    """ Make_geodesic(n)
    Makes the primitives of a geodesic surface based on an
    isocahedron which is subdivided n times in smaller triangles.
    Return the number of vertices, surfaces, associations and
    their related vectors.
    
    n: integer number of subdivisions (can be zero)
    
    >>> n_faces, n_vertices, myfaces, myvertices, myassoc = Make_geodesic(n)
    """
    support_code = """
    static int n_vertices;
    static int n_faces;
    static int n_edges;
    static float *vertices = NULL;
    static int *faces = NULL;
    static int *assoc = NULL;
    
    static int edge_walk; 
    static int *start = NULL; 
    static int *end = NULL; 
    static int *midpoint = NULL; 
    
    static void 
    init_icosahedron (void) 
    { 
        float t = (1+sqrt(5))/2;
        float tau = t/sqrt(1+t*t);
        float one = 1/sqrt(1+t*t);
        
        float icosahedron_vertices[] = 
        {tau, one, 0.0,
        -tau, one, 0.0,
        -tau, -one, 0.0,
        tau, -one, 0.0,
        one, 0.0 ,  tau,
        one, 0.0 , -tau,
        -one, 0.0 , -tau,
        -one, 0.0 , tau,
        0.0 , tau, one,
        0.0 , -tau, one,
        0.0 , -tau, -one,
        0.0 , tau, -one};
        
        int icosahedron_faces[] = 
        {4, 8, 7,
        4, 7, 9,
        5, 6, 11,
        5, 10, 6,
        0, 4, 3,
        0, 3, 5,
        2, 7, 1,
        2, 1, 6,
        8, 0, 11,
        8, 11, 1,
        9, 10, 3,
        9, 2, 10,
        8, 4, 0,
        11, 0, 5,
        4, 9, 3,
        5, 3, 10,
        7, 8, 1,
        6, 1, 11,
        7, 2, 9,
        6, 10, 2};
        
        n_vertices = 12;
        n_faces = 20;
        n_edges = 30;
        
        vertices = (float*)malloc(3*n_vertices*sizeof(float));
        faces = (int*)malloc(3*n_faces*sizeof(int));
        memcpy ((void*)vertices, (void*)icosahedron_vertices, 3*n_vertices*sizeof(float));
        memcpy ((void*)faces, (void*)icosahedron_faces, 3*n_faces*sizeof(int));
    }
    
    static int 
    search_midpoint (int index_start, int index_end) 
    { 
        int i;
        for (i=0; i<edge_walk; i++) 
            if ((start[i] == index_start && end[i] == index_end) || 
	        (start[i] == index_end && end[i] == index_start)) 
                {
	            int res = midpoint[i];
                
	            /* update the arrays */
	            start[i]    = start[edge_walk-1];
	            end[i]      = end[edge_walk-1];
	            midpoint[i] = midpoint[edge_walk-1];
	            edge_walk--;
	            
	            return res; 
                }
            
            /* vertex not in the list, so we add it */
            start[edge_walk] = index_start;
            end[edge_walk] = index_end; 
            midpoint[edge_walk] = n_vertices; 
            
            /* create new vertex */ 
            vertices[3*n_vertices]   = (vertices[3*index_start] + vertices[3*index_end]) / 2.0;
            vertices[3*n_vertices+1] = (vertices[3*index_start+1] + vertices[3*index_end+1]) / 2.0;
            vertices[3*n_vertices+2] = (vertices[3*index_start+2] + vertices[3*index_end+2]) / 2.0;
            
            /* normalize the new vertex */ 
            float length = sqrt (vertices[3*n_vertices] * vertices[3*n_vertices] +
        		       vertices[3*n_vertices+1] * vertices[3*n_vertices+1] +
        		       vertices[3*n_vertices+2] * vertices[3*n_vertices+2]);
            length = 1/length;
            vertices[3*n_vertices] *= length;
            vertices[3*n_vertices+1] *= length;
            vertices[3*n_vertices+2] *= length;
            
            n_vertices++;
            edge_walk++;
            return midpoint[edge_walk-1];
    }
    
    static void 
    subdivide (void) 
    { 
        int n_vertices_new = n_vertices+2*n_edges; 
        int n_faces_new = 4*n_faces; 
        int i; 
        
        edge_walk = 0;
        n_edges = 2*n_vertices + 3*n_faces; 
        start = (int*)malloc(n_edges*sizeof (int)); 
        end = (int*)malloc(n_edges*sizeof (int)); 
        midpoint = (int*)malloc(n_edges*sizeof (int)); 
        
        int *faces_old = (int*)malloc (3*n_faces*sizeof(int)); 
        faces_old = (int*)memcpy((void*)faces_old, (void*)faces, 3*n_faces*sizeof(int)); 
        vertices = (float*)realloc ((void*)vertices, 3*n_vertices_new*sizeof(float)); 
        faces = (int*)realloc ((void*)faces, 3*n_faces_new*sizeof(int)); 
        n_faces_new = 0; 
        
        for (i=0; i<n_faces; i++) 
        { 
            int a = faces_old[3*i]; 
            int b = faces_old[3*i+1]; 
            int c = faces_old[3*i+2]; 
            
            int ab_midpoint = search_midpoint (b, a);
            int bc_midpoint = search_midpoint (c, b);
            int ca_midpoint = search_midpoint (a, c);
            
            faces[3*n_faces_new] = a; 
            faces[3*n_faces_new+1] = ab_midpoint; 
            faces[3*n_faces_new+2] = ca_midpoint; 
            n_faces_new++; 
            faces[3*n_faces_new] = ca_midpoint; 
            faces[3*n_faces_new+1] = ab_midpoint; 
            faces[3*n_faces_new+2] = bc_midpoint; 
            n_faces_new++; 
            faces[3*n_faces_new] = ca_midpoint; 
            faces[3*n_faces_new+1] = bc_midpoint; 
            faces[3*n_faces_new+2] = c; 
            n_faces_new++; 
            faces[3*n_faces_new] = ab_midpoint; 
            faces[3*n_faces_new+1] = b; 
            faces[3*n_faces_new+2] = bc_midpoint; 
            n_faces_new++; 
        } 
        n_faces = n_faces_new; 
        free (start); 
        free (end); 
        free (midpoint); 
        free (faces_old); 
    } 
    
    static void 
    associativity (void) 
    { 
        //printf ("associativity 2\\n");
        int i;
        
        assoc = (int*)malloc(6*n_vertices*sizeof(int)); 
        
        for (int v=0; v<n_vertices; v++)
        {
            i = 0;
            for (int f=0; f<n_faces; f++)
            {
                if ((faces[3*f] == v) || (faces[3*f+1] == v) || (faces[3*f+2] == v)) {
                    assoc[6*v+i] = f;
                    i += 1;
                }
            }
            if (i==5) {
                assoc[6*v+i] = -99;
                i = 6;
            }
        }
    }
    
    static void 
    isocahedron (int n_subdivisions)
    {
        int i;
        
        init_icosahedron ();
        
        for (i=0; i<n_subdivisions; i++)
            subdivide ();
        
        associativity ();
    }
    
    static void
    free_memory ()
    {
        if (vertices) free (vertices);
        if (faces) free (faces);
        if (assoc) free (assoc);
    }
    """
    
    code = """
    isocahedron(n);
    
    //printf ( "\\nisocahedron 1 \\n" );
    //printf ( "long %zu\\n", sizeof(long) );
    //printf ( "myfaces %zu\\n", sizeof(myfaces(0,0)) );
    //printf ( "faces %zu\\n", sizeof(faces[0]) );
    //printf ( "myvertices %zu\\n", sizeof(myvertices(0,0)) );
    //printf ( "vertices %zu\\n", sizeof(vertices[0]) );
    
    for (int i=0; i<n_faces; i++) {
        myfaces(i,0) = faces[3*i];
        myfaces(i,1) = faces[3*i+1];
        myfaces(i,2) = faces[3*i+2];
    }
    
    for (int i=0; i<n_vertices; i++) {
        myvertices(i,0) = vertices[3*i];
        myvertices(i,1) = vertices[3*i+1];
        myvertices(i,2) = vertices[3*i+2];
        
        myassoc(i,0) = assoc[6*i];
        myassoc(i,1) = assoc[6*i+1];
        myassoc(i,2) = assoc[6*i+2];
        myassoc(i,3) = assoc[6*i+3];
        myassoc(i,4) = assoc[6*i+4];
        myassoc(i,5) = assoc[6*i+5];
    }
    
    free_memory();
    """
    n_faces = 20 * 4**n
    myfaces = numpy.empty((n_faces,3), dtype=numpy.int)
    n_vertices = 2 + 10 * 4**n
    myvertices = numpy.empty((n_vertices,3), dtype=numpy.float)
    myassoc = numpy.empty((n_vertices,6), dtype=numpy.int)
    get_axispos = scipy.weave.inline(code, ['n','myfaces','myvertices','myassoc'], type_converters=scipy.weave.converters.blitz, compiler='gcc', libraries=['m'], verbose=2, support_code=support_code, force=0)
    return n_faces, n_vertices, myfaces, myvertices, myassoc

def Match_assoc(faces, n_vertices):
    """
    Match_assoc(faces, n_vertices)
    
    Returns the list of faces associated with each vertice.
    There are 5 or 6 faces per vertice, if 5, the 6th is -99.
    
    >>> assoc = Match_assoc(faces, n_vertices)
    """
    code = """
    int ind = 0;
    for (int i=0; i<n_faces; i++) {
        for (int j=0; j<3; j++) {
            ind = faces(i, j);
            for (int k=0; k<6; k++) {
                if (assoc(ind, k)  == -99) {
                    assoc(ind, k) = i;
                    break;
                }
            }
        }
    }
    """
    n_faces = faces.shape[0]
    assoc = -99 * numpy.ones((n_vertices,6), dtype=numpy.int)
    get_assoc = scipy.weave.inline(code, ['n_faces','faces','assoc'], type_converters=scipy.weave.converters.blitz, compiler='gcc', libraries=['m'], verbose=2, force=0)
    return assoc

def Match_triangles(high_x, high_y, high_z, low_x, low_y, low_z):
    """Match_triangles(high_x, high_y, high_z, low_x, low_y, low_z)
    
    The idea is to identify the triangles of the high resolution tessellation
    that belong to the low resolution version. Because we use a subdivision
    algorithm, which splits each triangle into 4 smaller triangles, there 
    should be 4**(n_highres - n_lowres) triangles associated with each low
    resolution one.
    
    Returns the list of low resolution face indices associated with each
    high resolution one.
    
    >>> ind = Match_triangles(high_x, high_y, high_z, low_x, low_y, low_z)
    >>> n_lowres = ind.shape
    """
    code = """
    double dot, new_dot;
    
    for (int i=0; i<n_highres; i++) {
        dot = 0.;
        new_dot = 0.;
        for (int j=0; j<n_lowres; j++) {
            new_dot = high_x(i)*low_x(j) + high_y(i)*low_y(j) + high_z(i)*low_z(j);
            if (new_dot > dot) {
                dot = new_dot;
                ind(i) = j;
            }
        }
    }
    """
    n_highres = high_x.size
    n_lowres = low_x.size
    ind = numpy.zeros(n_highres, dtype='int')
    get_assoc = scipy.weave.inline(code, ['n_highres','n_lowres','ind', 'high_x', 'low_x', 'high_y', 'low_y', 'high_z', 'low_z'], type_converters=scipy.weave.converters.blitz, compiler='gcc', libraries=['m'], verbose=2, force=0)
    return ind

def Match_subtriangles(inds_highres, inds_lowres):
    """Match_subtriangles(inds_highres, inds_lowres)
    
    Given a list of match of triangles at one resolution (say 4 to 3)
    and another at a higher resolution (say 5 to 4), will match the 
    higher resolution with the base resolution (5 to 3).
    
    >>> ind = Match_subtriangles(inds_highres, inds_lowres)
    >>> inds_highres.shape = ind.shape
    """
    code = """
    int tmp_ind;
    
    for (int i=0; i<n_highres; i++) {
        tmp_ind = inds_highres(i);
        ind(i) = inds_lowres(tmp_ind);
    }
    """
    n_highres = inds_highres.size
    n_lowres = inds_lowres.size
    ind = numpy.zeros(n_highres, dtype='int')
    get_assoc = scipy.weave.inline(code, ['n_highres','n_lowres','ind', 'inds_highres', 'inds_lowres'], type_converters=scipy.weave.converters.blitz, compiler='gcc', libraries=['m'], verbose=2, force=0)
    return ind

def Radius(cosx, cosy, cosz, psi0, r, q, qp1by2om2):
    """Radius(cosx, cosy, cosz, psi0, r, q, qp1by2om2)
    >>>Radius(-1., 0., 0., 5454., 0.14, 56., 57./2)
    """
    code = """
    double x, y, z, rc2, rc, rx, rx3, psi, dpsi, dpsidx, dpsidy, dpsidz, dpsidr, dr;
    #line 100
    //std::cout << sizeof(r) << std::endl;
    #line 110
    int i = 0; // this will monitor the number of iterations
    int nmax = 50; // stop if convergence not reached
    //std::cout << "psi0: " << psi0 << std::endl;
    do {
        i += 1;
        if (i > nmax) {
            //std::cout << "Maximum iteration reached!!!" << std::endl;
            r = -99.99;
            break;
        }
        //std::cout << "i: " << i << std::endl;
        x = r*cosx;
        y = r*cosy;
        z = r*cosz;
        rc2 = x*x+y*y+z*z;
        rc = sqrt(rc2);
        rx = sqrt(rc2+1-2*x);
        rx3 = rx*rx*rx;
        psi = 1/rc + q/rx - q*x + qp1by2om2*(rc2-z*z);
        dpsi = -1/(rc*rc*rc)-q/rx3;
        dpsidx = x*(dpsi+2*qp1by2om2)+q*(1/rx3-1);
        dpsidy = y*(dpsi+2*qp1by2om2);
        dpsidz = z*dpsi;
        dpsidr = dpsidx*cosx+dpsidy*cosy+dpsidz*cosz;
        dr = (psi-psi0)/dpsidr;
        //std::cout << "psi: " << psi << ", dpsidr: " << dpsidr << std::endl;
        //std::cout << "r: " << r << ", dr: " << dr << std::endl;
        if ((r - dr) < 0.0) {
            r = 0.5 * r;
        } else {
            r = r - dr;
        }
    } while (fabs(dr) > 0.00001);
    return_val = r;
    """
    psi0 = numpy.float(psi0)
    r = numpy.float(r)
    q = numpy.float(q)
    cosx = numpy.float(cosx)
    cosy = numpy.float(cosy)
    cosz = numpy.float(cosz)
    qp1by2om2 = numpy.float(qp1by2om2)
    get_radius = scipy.weave.inline(code, ['r', 'cosx', 'cosy', 'cosz', 'psi0', 'q', 'qp1by2om2'], type_converters=scipy.weave.converters.blitz, compiler='gcc', verbose=2)
    r = get_radius
    return r

def Radii(cosx, cosy, cosz, psi0, r, q, qp1by2om2):
    """Radii(cosx, cosy, cosz, psi0, r, q, qp1by2om2)
    >>>Radii(numpy.array([-1.,0.,0.]), numpy.array([0.,0.1,0.1]), numpy.array([0.,0.,0.1]), 5454., 0.14, 56., 57./2)
    """
    code = """
    #pragma omp parallel shared(r,cosx,cosy,cosz,psi0,q,qp1by2om2,rout,n) default(none)
    {
    double x, y, z, rc2, rc, rx, rx3, psi, dpsi, dpsidx, dpsidy, dpsidz, dpsidr, dr;
    int ii; // this will monitor the number of iterations
    int nmax = 50; // stop if convergence not reached
    #pragma omp for
    for (int i=0; i<n; ++i) {
        ii = 0;
        rout(i) = r;
        do {
            ii += 1;
            if (ii > nmax) {
                //std::cout << "Maximum iteration reached!!!" << std::endl;
                rout(i) = -99.99;
                break;
            }
            x = rout(i)*cosx(i);
            y = rout(i)*cosy(i);
            z = rout(i)*cosz(i);
            rc2 = x*x+y*y+z*z;
            rc = sqrt(rc2);
            rx = sqrt(rc2+1-2*x);
            rx3 = rx*rx*rx;
            psi = 1/rc + q/rx - q*x + qp1by2om2*(rc2-z*z);
            dpsi = -1/(rc*rc*rc)-q/rx3;
            dpsidx = x*(dpsi+2*qp1by2om2)+q*(1/rx3-1);
            dpsidy = y*(dpsi+2*qp1by2om2);
            dpsidz = z*dpsi;
            dpsidr = dpsidx*cosx(i)+dpsidy*cosy(i)+dpsidz*cosz(i);
            dr = (psi-psi0)/dpsidr;
            if ((rout(i) - dr) < 0.0) {
                rout(i) = 0.5 * rout(i);
            } else {
                rout(i) = rout(i) - dr;
            }
        } while (fabs(dr) > 0.00001);
    }
    }
    """
    psi0 = numpy.float(psi0)
    r = numpy.float(r)
    q = numpy.float(q)
    qp1by2om2 = numpy.float(qp1by2om2)
    n = cosx.size
    rout = numpy.empty(n, dtype=float)
    try:
        if os.uname()[0] == 'Darwin':
            extra_compile_args = extra_link_args = ['-O3']
        else:
            extra_compile_args = extra_link_args = ['-O3 -fopenmp']
        get_radius = scipy.weave.inline(code, ['r', 'cosx', 'cosy', 'cosz', 'psi0', 'q', 'qp1by2om2', 'rout', 'n'], type_converters=scipy.weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=['<omp.h>'], verbose=2)
    except:
        get_radius = scipy.weave.inline(code, ['r', 'cosx', 'cosy', 'cosz', 'psi0', 'q', 'qp1by2om2', 'rout', 'n'], type_converters=scipy.weave.converters.blitz, compiler='gcc', extra_compile_args=['-O3'], extra_link_args=['-O3'], verbose=2)
    r = get_radius
    return rout

def Saddle(x, q, qp1by2om2):
    """Saddle(x, q, qp1by2om2)
    >>>Saddle(0.5, 56., 57./2)
    """
    code = """
        double rc, rx, rx3, dpsi, dpsidx, d2psidx2, dx;
        do {
            rc = fabs(x);
            rx = sqrt(rc*rc+1-2*x);
            rx3 = rx*rx*rx;
            dpsi = -1/(rc*rc*rc)-q/rx3;
            dpsidx = x*(dpsi+2*qp1by2om2)+q*(1/rx3-1);
            d2psidx2 = dpsi+3*(x*x/(rc*rc*rc*rc*rc)+q*(x-1)*(x-1)/(rx*rx*rx*rx*rx))+2*qp1by2om2;
            dx = -dpsidx/d2psidx2;
            x = x+dx;
        } while (fabs(dx/x) > 0.00001);
        return_val = x;
        """
    q = numpy.float(q)
    qp1by2om2 = numpy.float(qp1by2om2)
    get_saddle = scipy.weave.inline(code, ['x', 'q', 'qp1by2om2'], type_converters=scipy.weave.converters.blitz, compiler='gcc', verbose=2)
    x = get_saddle
    return x

