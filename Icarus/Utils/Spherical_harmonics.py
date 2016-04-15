# Licensed under a 3-clause BSD style license - see LICENSE

from .import_modules import *


##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##
## Contain functions to perform spherical harmonics
## calculations.
##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##


Norm_type = 1


def Alm(l,m,phi,theta,f):
    """Ylm(l,m,phi,theta,f)
    Returns the coefficient of the complex spherical harmonics
    for the pixelized function 'f'.
    
    l: Quantum number (0<=l).
    m: Quantum number (-l<=m<=l).
    phi: azimuth in the range [0,2*PI].
    theta: co-latitude in the range [0,PI].
    f: pixelized function f (f.size = phi.size = theta.size)
    """
    if Norm_type == 0: norm = 4*cts.PI
    elif Norm_type == 1: norm = 2*l+1.
    return (f * Ylm(phi,theta).conj()).real.sum() * norm/f.size

def Almr(l,m,phi,theta,f):
    """Ylmr(l,m,phi,theta,f)
    Returns the coefficient of the real spherical harmonics
    for the pixelized function 'f'.
    
    l: Quantum number (0<=l).
    m: Quantum number (-l<=m<=l).
    phi: azimuth in the range [0,2*PI].
    theta: co-latitude in the range [0,PI].
    f: pixelized function f (f.size = phi.size = theta.size)
    """
    if Norm_type == 0: norm = 4*cts.PI
    elif Norm_type == 1: norm = 2*l+1.
    return (f * Ylmr(l,m,phi,theta)).sum() * norm/f.size

def Composition(alm,phi,theta):
    """Composition(alm,phi,theta)
    Returns the pixelized function corresponding to the sum of
    the real spherical harmonics having the coefficients 'alm'.
    
    alm: Spherical harmonic coefficients. Must have the form:
        [A_{00},A_{1-1},A_{10},A_{11}, ]
    phi: azimuth in the range [0,2*PI].
    theta: co-latitude in the range [0,PI].
    
    >>> f = Composition(alm,phi,theta)
    """
    if isinstance(alm, (float, int)):
        alm = [alm]
    else:
        alm = list(alm[::-1]) # making the alm a list will be easier and reversing so that the first element pops first
    n = len(alm)
    lmax = np.sqrt(n).astype(int) - 1 # n = (lmax+1)**2
    f = 0.
    for l in xrange(lmax+1):
        for m in xrange(-l,l+1):
            f += alm.pop() * Ylmr(l,m,phi,theta)
    return f

def Decomposition(lmax,phi,theta,f,ndigit=None,norm=False):
    """Decomposition(lmax,phi,theta,f)
    Returns the coefficient of the real spherical harmonic
    decomposition of a pixelized function 'f' up to the quantum
    number l 'lmax', inclusive.
    
    lmax: Maximum order of decomposition for quantum number l.
    phi: azimuth in the range [0,2*PI].
    theta: co-latitude in the range [0,PI].
    f: pixelized function f (f.size = phi.size = theta.size)
    ndigit (None): if not None, will round off the results at
        ndigit (as per the np.round function).
    norm (False): if true, will normalize so that the sum of
        the square of the coefficients is unity.
    
    The returned array has the form:
    [A_{00},A_{1-1},A_{10},A_{11}, ,A_{lmax,-lmax}, ,A_{lmax,0}, ,A{lmax,lmax}]
    >>> alm = Decomposition(lmax,phi,theta,f)
    """
    alm = []
    for l in xrange(lmax+1):
        for m in xrange(-l,l+1):
            alm.append(Almr(l,m,phi,theta,f))
    alm = np.array(alm)
    if ndigit is not None:
        alm = np.round(alm, ndigit)
    if norm:
        alm /= np.sqrt((alm**2).sum())
    return alm

def Legendre_assoc(l,m,x):
    """Legendre_assoc(l,m,x)
    Associated Legendre polynomials normalized as in Ylm.
    
    l: Quantum number (0<=l).
    m: Quantum number (0<=m<=l).
    x: Argument, typically x=cos(theta) (abs(x)<=1).
    """
    l,m = int(l),int(m)
    assert 0<=m<=l and np.all(abs(x)<=1.)
    if Norm_type == 0: norm = np.sqrt(2*l+1) / np.sqrt(4*cts.PI)
    elif Norm_type == 1: norm = 1.
    if m == 0:
        pmm = norm * np.ones_like(x)
    else:
        pmm = (-1)**m * norm * Xfact(m) * (1-x**2)**(m/2.)
    if l == m:
        return pmm
    pmmp1 = x*pmm*np.sqrt(2*m+1)
    if l == m+1:
        return pmmp1
    for ll in xrange(m+2,l+1):
        pll = (x*(2*ll-1)*pmmp1 - np.sqrt((ll-1)**2 - m**2)*pmm)/np.sqrt(ll**2-m**2)
        pmm = pmmp1
        pmmp1 = pll
    return pll

def Normalization(val=1):
    """Normalization(val=1)
    Changes the global normalization factor.
    
    val (1):
        0 -> Implies that the spherical harmonic
        coefficients are equal to sqrt(2*l+1)/sqrt(4*PI)
        times the maximum/minimum value that the function
        can take.
        1 -> Implies that the spherical harmonic
        coefficients are equal to the maximum/minimum
        value that the function can take.
    """
    global Norm_type
    if val in [0,1]:
        Norm_type = val
    else:
        Norm_type = 1
    return

def Pretty_print_alm(alm, format=2):
    """Pretty_print_alm(alm, format=2)
    Returns a nice representation of the spherical harmonic
    coefficients.
    
    alm: Spherical harmonic coefficients. Must have the form:
        [A_{00},A_{1-1},A_{10},A_{11}, ]
    format (2):
        1: Inline format.
        2: Matrix format (diagonal is l,0; first off-diagonal
        is l,1; second off-diagonal is l,2; etc.).
    """
    n = len(alm)
    lmax = np.sqrt(n).astype(int) - 1 # n = (lmax+1)**2
    f = ''
    if format == 1:
        alm = list(alm[::-1]) # making the alm a list will be easier and reversing so that the first element pops first
        for l in xrange(lmax+1):
            for m in xrange(-l,l+1):
                f += 'l=%i,m=%s: %f ' %(l,m,alm.pop())
            f += '\n'
        print(f)
    elif format == 2:
        alm = list(alm[::-1]) # making the alm a list will be easier and reversing so that the first element pops first
        data = np.empty((lmax+1,lmax+1), dtype=float)
        for l in xrange(lmax+1):
            for m in xrange(-l,l+1):
                if m < 0:
                    data[l,l+m] = alm.pop()
                else:
                    data[l-m,l] = alm.pop()
        for l in xrange(lmax+1):
            for m in xrange(lmax+1):
                tmp = data[l,m]
                f += '% 4.4e | ' %(tmp)
            f = f[:-3]
            f += '\n'
        print(f)
    return

def Xfact(m):
    """Xfact(m)
    Computes (2m-1)!!/sqrt((2m)!)
    """
    res = 1.
    for i in xrange(1,2*m+1):
        if i % 2: res *= i # (2m-1)!!
        res /= np.sqrt(i) # sqrt((2m)!)
    return res

def Ylm(l,m,phi,theta):
    """Ylm(l,m,phi,theta)
    Returns the complex spherical harmonics.
    
    l: Quantum number (0<=l).
    m: Quantum number (-l<=m<=l).
    phi: azimuth in the range [0,2*PI].
    theta: co-latitude in the range [0,PI].
    """
    l,m = int(l),int(m)
    assert 0 <= abs(m) <=l
    if m > 0:
        return Legendre_assoc(l,m,np.cos(theta))*np.exp(1J*m*phi)
    elif m < 0:
        #return (-1)**m*Legendre_assoc(l,-m,np.cos(theta))*np.exp(1J*m*phi)
        return (-1)**m*Legendre_assoc(l,-m,np.cos(theta))*np.exp(1J*m*phi)
    return Legendre_assoc(l,m,np.cos(theta))*np.ones_like(phi)

def Ylmr(l,m,phi,theta):
    """Ylmr(l,m,phi,theta)
    Returns the real spherical harmonics.
    
    l: Quantum number (0<=l).
    m: Quantum number (-l<=m<=l).
    phi: azimuth in the range [0,2*PI].
    theta: co-latitude in the range [0,PI].
    """
    l,m = int(l),int(m)
    assert 0 <= abs(m) <=l
    if m > 0:
        return Legendre_assoc(l,m,np.cos(theta))*np.cos(m*phi)*np.sqrt(2)
    elif m < 0:
        #return (-1)**m*Legendre_assoc(l,-m,np.cos(theta))*np.sin(-m*phi)*np.sqrt(2)
        return Legendre_assoc(l,-m,np.cos(theta))*np.sin(m*phi)*np.sqrt(2)
    return Legendre_assoc(l,m,np.cos(theta))*np.ones_like(phi)





