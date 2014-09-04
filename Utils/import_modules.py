# Licensed under a 3-clause BSD style license - see LICENSE

# import general modules
import struct, getopt, sys
import os
from os import popen4, system, path
import glob
import types
import datetime
import operator
import string
import logging
import warnings

# import numpy and scipy modules
import scipy
import scipy.weave
import scipy.ndimage
import scipy.optimize
import scipy.interpolate
import scipy.io
import scipy.constants
import scipy.stats
import numpy
import numpy as np

try:
    import matplotlib, pylab
except:
    print( "Cannot import matplotlib/pylab. This is not a critical error, but some of the plotting functionalities might be impossible." )

# define some useful constants
cts = scipy.constants
cts.Msun = 1.9891e30 # kg
cts.Mjup = 1.8987e27 # kg
cts.Mearth = 5.9736e24 # kg
cts.Rsun = 695510000.0 # m
cts.Rjup = 71492000.0 # m
cts.Rearth = 6378000.0 # m
cts.g_earth = 9.80665 # m/s^2
cts.logg_earth = numpy.log10(cts.g_earth*100) # cgs
cts.g_sun = 27.94 * cts.g_earth # m/s^2
cts.logg_sun = numpy.log10(cts.g_sun*100) # cgs

# define more constants
cts.ARCSECTORAD = float('4.8481368110953599358991410235794797595635330237270e-6')
cts.RADTOARCSEC = float('206264.80624709635515647335733077861319665970087963')
cts.SECTORAD    = float('7.2722052166430399038487115353692196393452995355905e-5')
cts.RADTOSEC    = float('13750.987083139757010431557155385240879777313391975')
cts.RADTODEG    = float('57.295779513082320876798154814105170332405472466564')
cts.DEGTORAD    = float('1.7453292519943295769236907684886127134428718885417e-2')
cts.RADTOHRS    = float('3.8197186342054880584532103209403446888270314977710')
cts.HRSTORAD    = float('2.6179938779914943653855361527329190701643078328126e-1')
#cts.pi          = float('3.1415926535897932384626433832795028841971693993751')
#cts.twopi       = float('6.2831853071795864769252867665590057683943387987502')
cts.twopi       = cts.pi * 2
#PIBYTWO     = float('1.5707963267948966192313216916397514420985846996876')
cts.pibytwo       = cts.pi / 2
cts.SECPERDAY   = float('86400.0')
cts.SECPERJULYR = float('31557600.0')

# import some useful utility functions
from .Misc import Pprint



