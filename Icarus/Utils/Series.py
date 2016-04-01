# Licensed under a 3-clause BSD style license - see LICENSE

import sys
import os

import scipy.weave

try:
    from numba import autojit
except:
    print("Cannot load the numba module.")

from .import_modules import *

logger = logging.getLogger(__name__)


##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##
## Time series utilities
## Contain functions that pertain to "time series-related"
## purposes such as convolution, interpolation, rebinning, etc.
##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##


def Convolve_gaussian_tophat(arr, sigma=1., top=1):
    """
    Convolve an array with a Gaussian and a tophat
    function along the last dimension.

    arr (array): Array of values to be convolved.
    sigma (float): The width (sigma) of the Gaussian.
    top (int): The width of the tophat.

    Note: This function works on a multi-dimensional array
        but will only apply the convolution on the last
        axis (i.e. wavelength if it is a spectrum array).
    """
    ## We define the gaussian kernel
    m_gauss = int(4*sigma+0.5)
    w_gauss = 2*m_gauss+1
    k_gauss = np.exp(-0.5*(np.arange(w_gauss)-m_gauss)**2/sigma**2)
    ## We define the tophat kernel
    w_top = int(top)
    ## If the tophat's width is even, we need to center it so the width is odd in order to preserve the phase in the convolution
    if w_top%2 == 0:
        w_top += 1
        k_top = np.ones(w_top)
        k_top[0] = 0.5
        k_top[-1] = 0.5
    else:
        k_top = np.ones(w_top)
    ## Calculating the full kernel
    if w_gauss > w_top:
        kernel = scipy.ndimage.convolve1d(k_gauss, k_top, mode='constant', cval=0.0)
    else:
        kernel = scipy.ndimage.convolve1d(k_top, k_gauss, mode='constant', cval=0.0)
    ## Normalizing the kernel so the sum is unity
    kernel /= kernel.sum()
    ## Applying the kernel to the array of values
    newarr = scipy.ndimage.convolve1d(arr, kernel, axis=-1)
    return newarr

def Getaxispos_scalar(xold, xnew):
    """ Getaxispos_scalar(xold, xnew)
    Given a scalar xnew, returns the index and fractional weight
    that corresponds to the nearest linear interpolation from
    the vector xold.
    
    xold: vector of values to be interpolated from.
    xnew: scalar value to be interpolated.
    
    weight,index = Getaxispos_scalar(xold, xnew)
    """
    n = xold.shape[0]
    code = """
    int jl, ju, jm;
    double w;
    bool ascending = xold(n-1) > xold(0);
    jl = 0;
    ju = n;
    while ((ju-jl) > 1)
    {
        jm = (ju+jl)/2;
        if (ascending == (xnew > xold(jm)))
            jl = jm;
        else
            ju = jm;
    }
    jl = (jl < (n-1) ? jl : n-2);
    w = (xnew-xold(jl))/(xold(jl+1)-xold(jl));
    py::tuple results(2);
    results[0] = w;
    results[1] = jl;
    return_val = results;
    """
    xold = np.asarray(xold)
    xnew = float(xnew)
    get_axispos = scipy.weave.inline(code, ['xold', 'xnew', 'n'], type_converters=scipy.weave.converters.blitz, compiler='gcc', verbose=2)
    w,j = get_axispos
    return w,j

def Getaxispos_vector(xold, xnew):
    """ Getaxispos_vector(xold, xnew)
    Given a vector xnew, returns the indices and fractional weights
    that corresponds to their nearest linear interpolation from
    the vector xold.
    
    xold: vector of values to be interpolated from.
    xnew: vector of values to be interpolated.
    
    weights,indices = Getaxispos_scalar(xold, xnew)
    """
    logger.log(5, "start")
    xold = np.ascontiguousarray(xold)
    xnew = np.ascontiguousarray(xnew)
    n = xold.shape[0]
    m = xnew.shape[0]
    j = np.empty(m, dtype=int)
    w = np.empty(m, dtype=float)
    code = """
    #pragma omp parallel shared(xold,xnew,n,m,j,w) default(none)
    {
    int jl, ju, jm;
    double a;
    bool ascending = xold(n-1) > xold(0);
    #pragma omp for
    //std::cout << m << std::endl;
    for (int i=0; i<m; ++i) {
       //std::cout << i << " " << xnew(i) << std::endl;
       jl = 0;
       ju = n;
       while ((ju-jl) > 1)
       {
           //std::cout << "+++" << std::endl;
           //std::cout << jl << " " << ju << " " << jm << std::endl;
           jm = (ju+jl)/2;
           //std::cout << i << " " << xnew(i) << " " << xold(jm) << std::endl;
           if (ascending == (xnew(i) > xold(jm)))
               jl = jm;
           else
               ju = jm;
           //std::cout << jl << " " << ju << " " << jm << std::endl;
       }
       j(i) = (jl < (n-1) ? jl : n-2);
       w(i) = (xnew(i)-xold(j(i)))/(xold(j(i)+1)-xold(j(i)));
    }
    }
    """
    try:
        if os.uname()[0] == 'Darwin':
            extra_compile_args = extra_link_args = ['-O3']
        else:
            extra_compile_args = extra_link_args = ['-O3 -fopenmp']
        get_axispos = scipy.weave.inline(code, ['xold', 'xnew', 'n', 'm', 'j', 'w'], type_converters=scipy.weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=['<omp.h>'], verbose=2)
    except:
        get_axispos = scipy.weave.inline(code, ['xold', 'xnew', 'n', 'm', 'j', 'w'], type_converters=scipy.weave.converters.blitz, compiler='gcc', extra_compile_args=['-O3'], extra_link_args=['-O3'], verbose=2)
    tmp = get_axispos
    logger.log(5, "end")
    return w,j

def GPolynomial_fit(y, x=None, err=None, coeff=1, Xfnct=None, Xfnct_offset=False, chi2=True):
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
    chi2 (bool): If true, will also return the chi-square.

    Returns generalized polynomial coefficients
        shape (coeff)
        i.e. (a_n, a_(n-1), ..., a_1, a_0)
    """
    n = y.size
    if x is None:
        x = np.arange(n, dtype=float)
    if err is None:
        err = np.ones(n, dtype=float)
    elif np.size(err) == 1:
        err = np.ones(n, dtype=float)*err
    if Xfnct is None:
        Xfnct = np.ones(n, dtype=float)
    if Xfnct_offset:
        Xfnct_offset = 1
    else:
        Xfnct_offset = 0
    a = np.empty((n,coeff), dtype=float)
    b = np.empty(n, dtype=float)
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
    tmp = np.linalg.lstsq(a, b)
    if chi2:
        return tmp[0], tmp[1][0]
    return tmp[0]

def Interp_linear(y, weights, inds):
    """
    """
    code1d = """
    #pragma omp parallel shared(ynew, y, weights, inds) default(none)
    {
    double w1, w0;
    int j0, j1;
    #pragma omp for
    for (int i=0; i<nynew; i++) {
        w1 = weights(i);
        w0 = 1.-w1;
        j0 = inds(i);
        j1 = 1+j0;
        ynew(i) = y(j0)*w0 + y(j1)*w1;
    }
    }
    """
    code2d = """
    #pragma omp parallel shared(ynew, y, weights, inds) default(none)
    {
    double w1, w0;
    int j0, j1;
    #pragma omp for
    for (int j=0; j<n; j++) {
        for (int i=0; i<nynew; i++) {
            w1 = weights(i);
            w0 = 1.-w1;
            j0 = inds(i);
            j1 = 1+j0;
            ynew(j,i) = y(j,j0)*w0 + y(j,j1)*w1;
        }
    }
    }
    """
    y = np.asarray(y)
    weights = np.asarray(weights)
    inds = np.asarray(inds, dtype=int)
    nynew = weights.size
    if y.ndim == 1:
        ynew = np.empty(nynew, dtype=float)
        code = code1d
        args = ['y', 'ynew', 'weights', 'inds', 'nynew']
    elif y.ndim == 2:
        n = y.shape[0]
        ynew = np.empty((n,nynew), dtype=float)
        code = code2d
        args = ['y', 'ynew', 'weights', 'inds', 'nynew', 'n']
    else:
        print("Number of dimensions > 2 not supported!")
        return
    try:
        if os.uname()[0] == 'Darwin':
            extra_compile_args = extra_link_args = ['-O3']
        else:
            extra_compile_args = extra_link_args = ['-O3 -fopenmp']
        interp = scipy.weave.inline(code, args, type_converters=scipy.weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=['<omp.h>','<cmath>'], libraries=['m'], verbose=2)
    except:
        get_flux = scipy.weave.inline(code, args, type_converters=scipy.weave.converters.blitz, compiler='gcc', extra_compile_args=['-O3'], extra_link_args=['-O3'], headers=['<cmath>'], libraries=['m'], verbose=2)
    return ynew

def Interp_integrate(y, x, xnew):
    """
    Resample a time series (x,y) at the values xnew by
    performing an integration within each new bin of the
    old time series using the Euler method. Here we assume
    that the new time series is undersampling the old one.

    >>> x = np.arange(100.)
    >>> y = y = np.sin(x/10)
    >>> xnew = np.arange(20.)*5+0.3
    >>> ynew = Utils.Interp_integrate(y, x, xnew)
    """
    ynew = np.zeros_like(xnew)
    i = 0
    ii = 0
    while ii < xnew.size:
        weight = 0.
        val = 0.
        if ii == 0:
            xnewl = xnew[ii]-(xnew[ii+1]-xnew[ii])*0.5
        else:
            xnewl = (xnew[ii]+xnew[ii-1])*0.5
        if ii == xnew.size-1:
            xnewr = xnew[ii]+(xnew[ii]-xnew[ii-1])*0.5
        else:
            xnewr = (xnew[ii+1]+xnew[ii])*0.5
        while i < x.size:
            if i == 0:
                xl = x[i]-(x[i+1]-x[i])*0.5
            else:
                xl = (x[i]+x[i-1])*0.5
            if i == x.size-1:
                xr = x[i]+(x[i]-x[i-1])*0.5
            else:
                xr = (x[i+1]+x[i])*0.5
            ## Bin completely inside
            if xl >= xnewl and xr <= xnewr:
                weight += xr-xl
                val += y[i]*(xr-xl)
            ## Bin overlapping the right side
            elif xl < xnewr and xr > xnewr:
                weight += xnewr-xl
                val += y[i]*(xnewr-xl)
                ## Means we have to move to next xnew bin
                break
            ## Bin overlapping the left side
            elif xl < xnewl and xr > xnewl:
                weight += xr-xnewl
                val += y[i]*(xr-xnewl)
            ## Bin to the right
            elif xl >= xnewr:
                ## Means we are done
                break
            ## Bin to the left
            elif xr <= xnewl:
                pass
            ## This condition should not happen
            else:
                pass
            i += 1
        ## Add the sum to ynew
        if weight != 0:
            ynew[ii] = val/weight
        ii += 1
    return ynew

if 'numba' in sys.modules:
    Interp_integrate = autojit(Interp_integrate)

def Resample_linlog(xold):
    """Resample_linlog(xold)
    Resample a linear wavelength vector to a log space and
    returns the new vector and the Doppler shift z.
    
    The resampling is done such that the largest wavelength interval
    is conserved in order to preserve the spectral resolution.
    
    The Doppler shift is:
        1+z = lambda_1 / lambda_0

    In the non-relativistic limit:
        z = v/c
    
    >>> xnew, z = Resample_linlog(xold)
    """
    z = xold[-2] / xold[-1] - 1
    ## The number of data points to cover the spectal range is
    n = np.ceil( np.log(xold[0]/xold[-1]) / np.log(1+z) ) + 1
    xnew = xold[-1] * (1+z)**np.arange(n)[::-1]
    return xnew, np.abs(z)

def Resample_loglin(xold):
    """Resample_loglin(xold)
    Resample a log wavelength vector to a linear space.
    
    The resampling is done such that the smallest wavelength interval
    is conserved in order to preserve the spectral resolution.
    
    >>> xnew = Resample_loglin(xold)
    """
    step = xold[1] - xold[0]
    xnew = np.arange(xold[0], xold[-1]+step, step)
    return xnew


