# Licensed under a 3-clause BSD style license - see LICENSE

## import general modules
#import struct, getopt, sys, types, datetime, operator, string, warnings
#import os
#import glob
import logging

## import numpy and scipy modules
import numpy as np
import scipy
import scipy.optimize
import scipy.interpolate
import scipy.constants
import scipy.stats
#import scipy.weave
#import scipy.ndimage
#import scipy.io

try:
    import matplotlib, pylab
    HAS_MATPLOTLIB = True
    try:
        import seaborn as sns
        HAS_SEABORN = True
        sns.set_style('ticks')
        sns.set_context(font_scale=1.0, rc={'lines.markeredgewidth': 1.0})
    except:
        HAS_SEABORN = False
        print( "Cannot import seaborn. This is not a critical error, but you might want to consider in order to have nicer plots." )
except:
    HAS_MATPLOTLIB = False
    HAS_SEABORN = False
    print( "Cannot import matplotlib/pylab. This is not a critical error, but some of the plotting functionalities might be impossible." )

## define some useful constants
cts = scipy.constants
cts.Msun = 1.9891e30 # kg
cts.Mjup = 1.8987e27 # kg
cts.Mearth = 5.9736e24 # kg
cts.Rsun = 695510000.0 # m
cts.Rjup = 71492000.0 # m
cts.Rearth = 6378000.0 # m
cts.g_earth = 9.80665 # m/s^2
cts.logg_earth = np.log10(cts.g_earth*100) # cgs
cts.g_sun = 27.94 * cts.g_earth # m/s^2
cts.logg_sun = np.log10(cts.g_sun*100) # cgs

## define more constants
cts.ARCSECTORAD = float('4.8481368110953599358991410235794797595635330237270e-6')
cts.RADTOARCSEC = float('206264.80624709635515647335733077861319665970087963')
cts.SECTORAD    = float('7.2722052166430399038487115353692196393452995355905e-5')
cts.RADTOSEC    = float('13750.987083139757010431557155385240879777313391975')
cts.RADTODEG    = float('57.295779513082320876798154814105170332405472466564')
cts.DEGTORAD    = float('1.7453292519943295769236907684886127134428718885417e-2')
cts.RADTOHRS    = float('3.8197186342054880584532103209403446888270314977710')
cts.HRSTORAD    = float('2.6179938779914943653855361527329190701643078328126e-1')
cts.PI          = cts.pi
cts.TWOPI       = cts.PI * 2
cts.PIBYTWO     = cts.PI / 2
cts.SECPERDAY   = float('86400.0')
cts.SECPERJULYR = float('31557600.0')

## import some useful utility functions
from .Misc import Pprint



