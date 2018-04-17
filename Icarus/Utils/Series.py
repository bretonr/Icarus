# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function, division

import sys
import os

try:
    from scipy import weave 
except:
    try:
        import weave
    except:
        print('weave cannot be import from scipy nor on its own.')

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

def Doppler_shift_spectrum(fref, wref, wobs, v):
    """
    Simple Doppler shifting of a spectrum using a linear interpolation.

    This Doppler shifting takes into account the Doppler boosting
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
    fref : ndarray
        Rest flux in energy per unit time per unit solid angle per unit
        wavelength.
    wref : ndarray
        Rest wavelengths
    wobs : ndarray
        Wavelengths to be interpolated at
    v : float
        Velocity in v/c unit
            Positive velocity: blueshift
            Negative velocity: redshift

    Returns
    -------
    fobs : ndarray
        Doppler shifted and boosted spectrum.
    """
    logger.log(5, "start")
    wref = np.ascontiguousarray(wref, dtype=float)
    fref = np.ascontiguousarray(fref, dtype=float)
    wobs = np.ascontiguousarray(wobs, dtype=float)
    v = np.float(v)
    nref = wref.size
    nobs = wobs.size
    fobs = np.empty(nobs, dtype=float)
    code = """
    #pragma omp parallel shared(wref,wobs,fref,fobs,nref,nobs,v) default(none)
    {
    int jl, ju, jm, j;
    double w, wav;
    bool ascending = wref(nref-1) > wref(0);
    #pragma omp for
    //std::cout << nobs << std::endl;
    for (int i=0; i<nobs; ++i) {
        wav = wobs(i)*(1+v);
        //std::cout << i << " " << wav << std::endl;
        jl = 0;
        ju = nref;
        while ((ju-jl) > 1)
        {
            //std::cout << "+++" << std::endl;
            //std::cout << jl << " " << ju << " " << jm << std::endl;
            jm = (ju+jl)/2;
            //std::cout << i << " " << wav << " " << wref(jm) << std::endl;
            if (ascending == (wav > wref(jm)))
                jl = jm;
            else
                ju = jm;
            //std::cout << jl << " " << ju << " " << jm << std::endl;
        }
        j = (jl < (nref-1) ? jl : nref-2);
        w = (wav-wref(j))/(wref(j+1)-wref(j));
        fobs(i) = (fref(j)*(1-w) + fref(j+1)*w) * (1+5*v);
    }
    }
    """
    if os.uname()[0] == 'Darwin':
        extra_compile_args = extra_link_args = ['-O3']
        headers = ['<cmath>']
    else:
        extra_compile_args = extra_link_args = ['-O3 -fopenmp']
        headers = ['<omp.h>','<cmath>']
    get_axispos = weave.inline(code, ['wref', 'wobs', 'fref', 'fobs', 'nref', 'nobs', 'v'], type_converters=weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=headers, verbose=2)
    tmp = get_axispos
    logger.log(5, "end")
    return fobs

def Doppler_shift_spectrum_integrate(fref, wobs, v, refstart, refstep):
    """
    Takes a reference spectrum, Doppler shifts it, and calculate
    the new spectral flux values at the provided observed wavelengths.

    - Assumes constant bin size and separation for the reference spectrum.
    - Assumes that the observed spectrum bin size is larger than
        the reference spectrum bin size and performs the integration.
        If it was smaller, a simple interpolation would be enough.
    - Takes into account the Doppler boosting component. I_nu/nu^3 is a
        Lorentz invariant (and hence I_lambda/nu^5).
        In this case, we have F_lambda and so the boosting is
            F(lambda) = F(lambda') * (1+5v/c)

            (see Doppler_shift_spectrum for a full explanation)

    fref: reference flux values
    wobs: observed wavelengths
    v: Doppler velocity shift (in m/s)
    refstart: wavelength of the first reference spectrum data point
    refstep: wavelength step size of the reference spectrum

    N.B. Could be optimized for the case of constant binning for the
    observed spectrum.
    """
    nobs = wobs.size
    nref = fref.size
    fbin = np.zeros(nobs, dtype=float)
    fref = np.ascontiguousarray(fref, dtype=float)
    wobs = np.ascontiguousarray(wobs, dtype=float)
    v = np.float(v)
    refstart = np.float(refstart)
    refstep = np.float(refstep)
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
    //double scale = sqrt( (1.+v/299792458.0)/(1.-v/299792458.0) ); // this is the Doppler scaling factor for the observed wavelength
    double scale = 1.+v;
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
            fbin(n) *= (1+5*v);
            //std::cout << "fbin(n)4: " << fbin(n) << std::endl;
        }
    }
    """
    rebin = weave.inline(code, ['refstart', 'refstep', 'nobs', 'nref', 'fref', 'fbin', 'wobs', 'v'], type_converters=weave.converters.blitz, compiler='gcc', libraries=['m'])
    tmp = rebin
    return fbin

def FFTConvolve1D(in1, in2, axis=-1):
    """
    Convolve a N-dimensional array with a one dimensional kernel using FFT
    along a specified axis.

    Parameters
    ----------
    in1 : ndarray
        Input array to operate the convolution on. Can be any dimension.
    in2 : ndarray
        Input convolution kernel. Must be 1-dimensional.
        The dimension of in2 must much the axis dimension of in1 over which the
        convolution is performed.
    axis : int
        Axis over which the convolution is performed.

    Returns
    -------
    convarr : ndarray
        Convolved array having the same dimensions as in1. Note that the
        convolution implicitely uses the "same" method, applied to in1.
    """
    ## Making sure that the dimensions match
    #if in1.shape[axis] != in2.size:
    #    raise ValueError("The 'axis' dimension of in1 must match the size of in2")

    ## Formatting the kernel to match the input array
    in2 = in2.copy()
    s2 = np.ones(in1.ndim, dtype=int)
    s2[axis] = in2.size
    in2.shape = s2

    ## Working out the size of the convolution array and the slice to extract
    size = in1.shape[axis] + in2.size - 1
    fftslice = [slice(l) for l in in1.shape]
    fftslice[axis] = slice(0, int(size))
    fftslice = tuple(fftslice)
    
    ## Using 2**n FFT for speed
    fftsize = 2**int(np.ceil(np.log2(size)))

    ## Applying the convolution theorem in the Fourier space
    fftarr = scipy.fftpack.fft(in1, fftsize, axis=axis)
    fftarr *= scipy.fftpack.fft(in2, fftsize, axis=axis)
    convarr = scipy.fftpack.ifft(fftarr, axis=axis)[fftslice].copy()

    return scipy.signal.signaltools._centered(convarr, in1.shape)

def Getaxispos_scalar(xold, xnew):
    """
    Given a scalar xnew, returns the index and fractional weight
    that corresponds to the nearest linear interpolation from
    the vector xold.

    xold: vector of values to be interpolated from.
    xnew: scalar value to be interpolated.

    weight,index = Getaxispos_scalar(xold, xnew)
    """
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
    xold = np.ascontiguousarray(xold, dtype=float)
    xnew = np.float(xnew)
    n = xold.shape[0]
    get_axispos = weave.inline(code, ['xold', 'xnew', 'n'], type_converters=weave.converters.blitz, compiler='gcc', verbose=2)
    w,j = get_axispos
    return w,j

def Getaxispos_vector(xold, xnew):
    """
    Given a vector xnew, returns the indices and fractional weights
    that corresponds to their nearest linear interpolation from
    the vector xold.

    xold: vector of values to be interpolated from.
    xnew: vector of values to be interpolated.

    weights,indices = Getaxispos_scalar(xold, xnew)
    """
    logger.log(5, "start")
    xold = np.ascontiguousarray(xold, dtype=float)
    xnew = np.ascontiguousarray(xnew, dtype=float)
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
    if os.uname()[0] == 'Darwin':
        extra_compile_args = extra_link_args = ['-O3']
        headers = ['<cmath>']
    else:
        extra_compile_args = extra_link_args = ['-O3 -fopenmp']
        headers = ['<omp.h>','<cmath>']
    get_axispos = weave.inline(code, ['xold', 'xnew', 'n', 'm', 'j', 'w'], type_converters=weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=headers, verbose=2)
    tmp = get_axispos
    logger.log(5, "end")
    return w,j

def General_polynomial_fit(y, x=None, err=None, coeff=1, Xfnct=None, Xfnct_offset=False, chi2=True):
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
    y = np.ascontiguousarray(y, dtype=float)
    n = y.size
    if x is None:
        x = np.arange(n, dtype=float)
    else:
        x = np.ascontiguousarray(x, dtype=float)
    if err is None:
        err = np.ones(n, dtype=float)
    elif np.size(err) == 1:
        err = np.ones(n, dtype=float)*err
    else:
        err = np.ascontiguousarray(err, dtype=float)
    if Xfnct is None:
        Xfnct = np.ones(n, dtype=float)
    else:
        Xfnct = np.ascontiguousarray(Xfnct, dtype=float)
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
    prep_lstsq = weave.inline(code, ['y', 'x', 'err', 'Xfnct', 'Xfnct_offset', 'a', 'b', 'n', 'coeff'], type_converters=weave.converters.blitz, compiler='gcc')
    tmp = prep_lstsq
    tmp = np.linalg.lstsq(a, b)
    if chi2:
        return tmp[0], tmp[1][0]
    return tmp[0]

def Interp_linear(y, x, xnew):
    """
    Given a vector xnew, returns the indices and fractional weights
    that corresponds to their nearest linear interpolation from
    the vector xold.

    y: y variables to be interpolated from.
    x: x variables to be interpolated from.
    xnew: x variables to be interpolated at.

    weights,indices = Getaxispos_scalar(xold, xnew)
    """
    logger.log(5, "start")
    x = np.ascontiguousarray(x, dtype=float)
    y = np.ascontiguousarray(y, dtype=float)
    xnew = np.ascontiguousarray(xnew, dtype=float)
    n_old = x.size
    n_new = xnew.size
    ynew = np.empty(n_new, dtype=float)
    code = """
    #pragma omp parallel shared(x,xnew,y,ynew,n_old,n_new) default(none)
    {
    int jl, ju, jm, j;
    double w;
    bool ascending = x(n-1) > x(0);
    #pragma omp for
    //std::cout << n_new << std::endl;
    for (int i=0; i<n_new; ++i) {
       //std::cout << i << " " << xnew(i) << std::endl;
       jl = 0;
       ju = n_old;
       while ((ju-jl) > 1)
       {
           //std::cout << "+++" << std::endl;
           //std::cout << jl << " " << ju << " " << jm << std::endl;
           jm = (ju+jl)/2;
           //std::cout << i << " " << xnew(i) << " " << x(jm) << std::endl;
           if (ascending == (xnew(i) > x(jm)))
               jl = jm;
           else
               ju = jm;
           //std::cout << jl << " " << ju << " " << jm << std::endl;
       }
       j = (jl < (n_old-1) ? jl : n_old-2);
       w = (xnew(i)-x(j))/(x(j+1)-x(j));
       ynew(i) = y(j)*(1-w) + y(j+1)*w;
    }
    }
    """
    if os.uname()[0] == 'Darwin':
        extra_compile_args = extra_link_args = ['-O3']
        headers = ['<cmath>']
    else:
        extra_compile_args = extra_link_args = ['-O3 -fopenmp']
        headers = ['<omp.h>','<cmath>']
    get_axispos = weave.inline(code, ['x', 'xnew', 'y', 'ynew', 'n_old', 'n_new'], type_converters=weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=headers, verbose=2)
    tmp = get_axispos
    logger.log(5, "end")
    return ynew

def Interp_linear2(y, weights, inds):
    """
    Given some weights and indices (from Getaxispos), evaluate the linear
    interpolation of the original time series.

    >>> x = np.arange(100.)
    >>> y = np.sin(x/10)
    >>> xnew = np.arange(20.)*5+0.3
    >>> weights,indices = Getaxispos_scalar(x, xnew)
    >>> ynew = Utils.Interp_integrate(y, weights, indices)
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
    y = np.ascontiguousarray(y, dtype=float)
    weights = np.ascontiguousarray(weights, dtype=float)
    inds = np.ascontiguousarray(inds, dtype=int)
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
    if os.uname()[0] == 'Darwin':
        extra_compile_args = extra_link_args = ['-O3']
        headers = ['<cmath>']
    else:
        extra_compile_args = extra_link_args = ['-O3 -fopenmp']
        headers = ['<omp.h>','<cmath>']
    interp = weave.inline(code, args, type_converters=weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=headers, libraries=['m'], verbose=2)
    return ynew

def Interp_linear_integrate(y, x, xnew):
    """
    Resample a time series (x,y) at the values xnew by performing an
    integration within each new bin of the old time series using the Euler
    method. Here we assume that the new time series is undersampling the old
    one, otherwise it is just equivalent to linearly interpolating.

    Parameters
    ----------
    y : (N,...) ndarray
        y values to interpolate from. The array can be multi-dimensional. The
        interpolation will be carried along the first axis.
    x : (N,) ndarray
        x values to interpolate from. y = f(x)
    xnew : (M,) ndarray
        x values to interpolate at.

    Return
    ------
    ynew : (M,...) ndarray
        y values interpolated at. The first dimension is the same as xnew,
        while the other dimensions, if any, will match the other dimensions of
        x.

    >>> x = np.arange(100.)
    >>> y = np.sin(x/10)
    >>> xnew = np.arange(20.)*5+0.3
    >>> ynew = Interp_linear_integrate(y, x, xnew)
    """
    shape = list(y.shape)
    shape[0] = xnew.size
    ynew = np.zeros(shape, dtype=float)
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
    Interp_linear_integrate = autojit(Interp_linear_integrate)

def Resample_linlog(xold):
    """
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
    """
    Resample a log wavelength vector to a linear space.

    The resampling is done such that the smallest wavelength interval
    is conserved in order to preserve the spectral resolution.

    >>> xnew = Resample_loglin(xold)
    """
    step = xold[1] - xold[0]
    xnew = np.arange(xold[0], xold[-1]+step, step)
    return xnew



