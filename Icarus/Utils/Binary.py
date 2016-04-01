# Licensed under a 3-clause BSD style license - see LICENSE

import os

import scipy.weave

from .import_modules import *

logger = logging.getLogger(__name__)


##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##
## Binary utilities
## Contain functions that pertain "binary-related" purposes
## such as orbital separation, mass ratio, etc.
##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##


def Err_velocity(chi2, vels, dof, clip=None, normalize=False, redchi2_unity=False, verbose=False):
    """
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
    p_n = np.poly1d(a_n)
    vfit = -0.5*a_n[1]/a_n[0]
    chi2fit = p_n(vfit)
    err_vfit = np.sqrt(1/a_n[0])
    if redchi2_unity:
        err_vfit *= np.sqrt(chi2fit/dof)
    if verbose:
        plotxy(chi2, vels, symbol=3)
        plotxy(y, x, line=None, symbol=3, color=4)
        plotxy(p_n(vels), vels, color=2)
    return chi2fit, vfit, err_vfit

def Get_potential(x, y, z, q, omega=1.):
    qp1by2om2 = (q+1.)/2.*omega**2
    rc2 = x**2+y**2+z**2
    rx = np.sqrt(rc2+1-2*x)
    rc = np.sqrt(rc2)
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

def Get_K_to_q(porb, asini):
    """
    Returns the K_to_q conversion factor given an
    orbital period and an observed asini.
    The K is for the "primary" whereas asini is that
    of the "secondary" star.
    
    porb: Orbital period in seconds.
    asini: Projected semi-major axis in lt-s.
    
    >>> K_to_q = Get_K_to_q(porb, asini)
    """
    return porb / (cts.TWOPI * asini * cts.c)

def Mass_companion(mass_function, q, incl):
    """
    Returns the mass of the neutron star companion.
    
    mass_function: Mass function of the neutron star.
    q: Mass ratio (q = M_ns/M_comp)
    incl: Orbital inclination in radians.
    
    >>> Mass_companion(mass_function, q, incl)
    """
    return mass_function * (1+q)**2 / np.sin(incl)**3

def Mass_function(asini, porb):
    """
    Returns the mass function of an orbit.

    asini: Projected semi-major axis in lt-s.
    porb: Orbital period in s.
    """
    mfunc = 8015123.37129 * asini**3 / porb**2
    return mfunc

def Mass_function2(M_ns, M_comp, incl):
    """
    Returns the mass function of an orbit.

    M_ns: Mass of the neutron star in solar mass.
    M_comp: Mass of the companion in solar mass.
    incl: Orbital inclination in radians.
    """
    mfunc = 8015123.37129 * asini**3 / porb**2
    return mfunc

def Mass_ratio(mass_function, M_ns, incl):
    """
    Returns the mass ratio of a binary (q = M_ns/M_comp).
    
    mass_function: Mass function of the neutron star.
    M_ns: Neutron star mass in solar mass.
    incl: Orbital inclination in radians.
    
    >>> Mass_ratio(mass_function, M_ns, incl)
    """
    q = -1./9
    r = 0.5*M_ns*np.sin(incl)**3/mass_function + 1./27
    s = (r + np.sqrt(q**3+r**2))**(1./3)
    t = (r - np.sqrt(q**3+r**2))**(1./3)
    return s+t-2./3

def Mass_ratio2(M_ns, porb, incl, k1):
    """
    Returns the mass ratio of a binary (q = M_ns/M_comp).

    M_ns: Neutron star mass in solar mass.
    porb: Orbital period in s.
    incl: Orbital inclination in radians.
    k1: Projected orbital velocity in m/s.


    Note: From Kepler's law
    M2 = \frac{P}{2\pi G} (\frac{1+q}{q})^2 \frac{K1^3}{\sin^3 i}
    with q = M2/M1

    Solve q from the equation:
    (A-1)q^2 + 2Aq +A = 0
    with A = \frac{P}{2\pi G} \frac{K1^3}{M2 \sin^3 i}

    q = (-A \pm \sqrt(A)) / (A-1) = (-A - \sqrt(A)) / (A-1)
    but only the negative solution is possible given than A > 0.
    Also, q is positive only for A < 1.
    """
    ## Calculate the right-hand side value
    A = porb / (cts.TWOPI*cts.G*cts.Msun * M2) * k1**3 / np.sin(incl)**3
    if A >= 1:
        print("No valid solution for q!")
    q = (-A - np.sqrt(A)) / (A-1)
    return q

def Orbital_separation(asini, q, incl):
    """
    Returns the orbital separation in m.
    
    asini: Projected semi-major axis of the neutron star in lt-s.
    q: Mass ratio of the binary (q = M_ns/M_comp).
    incl: Orbital inclination in radians.
    
    >>> Orbital_separation(asini, q, incl)
    """
    return asini*(1+q)/np.sin(incl)*cts.c

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
    rx = np.sqrt(rc2+1-2*x)
    rc = np.sqrt(rc2)
    psi = 1/rc + q/rx - q*x + qp1by2om2*(rc2-z**2)
    dpsi = -1/rc**3-q/rx**3
    dpsidx = x*(dpsi+2*qp1by2om2)+q*(-1+1/rx**3)
    dpsidy = y*(dpsi+2*qp1by2om2)
    dpsidz = z*dpsi
    return rc, rx, dpsi, dpsidx, dpsidy, dpsidz, psi

def Radius(cosx, cosy, cosz, psi0, r, q, qp1by2om2):
    """
    >>>Radius(-1., 0., 0., 5454., 0.14, 56., 57./2)
    """
    logger.log(9, "start")
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
    psi0 = np.float(psi0)
    r = np.float(r)
    q = np.float(q)
    cosx = np.float(cosx)
    cosy = np.float(cosy)
    cosz = np.float(cosz)
    qp1by2om2 = np.float(qp1by2om2)
    get_radius = scipy.weave.inline(code, ['r', 'cosx', 'cosy', 'cosz', 'psi0', 'q', 'qp1by2om2'], type_converters=scipy.weave.converters.blitz, compiler='gcc', verbose=2)
    r = get_radius
    logger.log(9, "end")
    return r

def Radii(cosx, cosy, cosz, psi0, r, q, qp1by2om2):
    """
    >>>Radii(np.array([-1.,0.,0.]), np.array([0.,0.1,0.1]), np.array([0.,0.,0.1]), 5454., 0.14, 56., 57./2)
    """
    logger.log(9, "start")
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
    cosx = np.ascontiguousarray(cosx)
    cosy = np.ascontiguousarray(cosy)
    cosz = np.ascontiguousarray(cosz)
    psi0 = np.float(psi0)
    r = np.float(r)
    q = np.float(q)
    qp1by2om2 = np.float(qp1by2om2)
    n = cosx.size
    rout = np.empty(n, dtype=float)
    try:
        if os.uname()[0] == 'Darwin':
            extra_compile_args = extra_link_args = ['-O3']
        else:
            extra_compile_args = extra_link_args = ['-O3 -fopenmp']
        get_radius = scipy.weave.inline(code, ['r', 'cosx', 'cosy', 'cosz', 'psi0', 'q', 'qp1by2om2', 'rout', 'n'], type_converters=scipy.weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=['<omp.h>'], verbose=2)
    except:
        get_radius = scipy.weave.inline(code, ['r', 'cosx', 'cosy', 'cosz', 'psi0', 'q', 'qp1by2om2', 'rout', 'n'], type_converters=scipy.weave.converters.blitz, compiler='gcc', extra_compile_args=['-O3'], extra_link_args=['-O3'], verbose=2)
    r = get_radius
    logger.log(9, "end")
    return rout

def Roche_lobe(q):
    """
    Calculates the volume-averaged Roche lobe radius using the Eggleton
    approximation.

    RL = 0.49*q**(2./3) / (0.6*q**(2./3) + ln(1+q**(1./3)))

    Parameters
    ----------
    q : float, array
        Mass ratio of the system, q=M1/M2, where source 1 is the one under
        consideration.

    Outputs
    -------
    RL : float, array
        Roche lobe radius in units in orbital separation.
    """
    RL = 0.49*q**(2./3) / (0.6*q**(2./3) + np.log(1+q**(1./3)))
    return RL

def Saddle(x, q, qp1by2om2):
    """
    >>>Saddle(0.5, 56., 57./2)
    """
    logger.log(9, "start")
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
    q = np.float(q)
    qp1by2om2 = np.float(qp1by2om2)
    get_saddle = scipy.weave.inline(code, ['x', 'q', 'qp1by2om2'], type_converters=scipy.weave.converters.blitz, compiler='gcc', verbose=2)
    x = get_saddle
    logger.log(9, "end")
    return x


