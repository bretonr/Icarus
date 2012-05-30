#!/usr/bin/env python
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

# define some useful constants
cts = scipy.constants
cts.Msun = 1.9891e30
cts.Mjup = 1.8987e27
cts.Mearth = 5.9736e24
cts.Rsun = 695510000.0
cts.Rjup = 71492000.0
cts.Rearth = 6378000.0
cts.g_earth = 9.80665
cts.logg_earth = numpy.log10(cts.g_earth*100)
cts.g_sun = 27.94 * cts.g_earth
cts.logg_sun = numpy.log10(cts.g_sun*100)





