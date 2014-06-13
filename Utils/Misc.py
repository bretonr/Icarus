# Licensed under a 3-clause BSD style license - see LICENSE

from .import_modules import *


##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##
## Miscellaneous utilities
## Contain functions that do not pertain to a particular class.
##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##


def Fit_linear(y, x=None, err=1.0, m=None, b=None, output=None, inline=False):
    """
    Fit_linear(y, x=None, err=1.0, m=None, b=None, output=None, inline=False):
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

def Sort_list(lst, cols):
    """Sort_list(lst, cols)
    Sorts inplace a list by multiple columns.
    
    lst: List to be sorted.
    cols: Columns to be sorted, cols[0] first,
        cols[1] second, etc.
    
    >>> lst = [(1,2,4),(3,2,1),(2,2,2),(2,1,4),(2,4,1)]
    >>> Sort_list(lst, [2,1])
    """
    from operator import itemgetter
    for keycolumn in reversed(cols):
        lst.sort(key=itemgetter(keycolumn))
    return


