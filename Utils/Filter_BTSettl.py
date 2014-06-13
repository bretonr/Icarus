# Licensed under a 3-clause BSD style license - see LICENSE

from .import_modules import *
from . import Filter
try:
    #import h5py
    pass
except:
    print( "Failed at importing h5py. Some data reading functionalities might not work properly." )


##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##
## Contain functions to process BT-Settl AGSS2009 spectra
##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##

def Bandint_BTSettl_AGSS2009(atmo_flns, band_fln, output_fln, nu=True):
    """
    Integrate BTSettl AGSS2009 atmosphere models (HDF format) through a
    given filter.
    
    atmo_flns: atmosphere model filenames (HDF format).
    band_fln: filter filename.
    output_fln: output filename.
    nu (True): Whether the integration should be performed in the frequency
        (STMAG system) or wavelength (ABMAG system) space.
    """
    # Load the filter interpolator
    band_func = Filter.Load_filter(band_fln, nu=nu)
    
    # Iterating through the atmosphere data to integrate through the filter
    atmo_flns.sort()
    with open(output_fln, 'w') as output:
        for atmo_fln in atmo_flns:
            print( 'Integrating %s' %atmo_fln )
            ind = atmo_fln.find('lte')
            temp = float(atmo_fln[ind+3:ind+6])*100
            logg = float(atmo_fln[ind+7:ind+10])
            if atmo_fln[ind+6] == '+':
                logg *= -1
            w, f = Load_BTSettl_AGSS2009(atmo_fln, nu=nu)
            flux = Filter.Band_integration(band_func, w, f, nu=nu)
            output.write( '%7.1f %4.2f %e\n' %(temp,logg,flux) )
    return

def Doppler_boosting_BTSettl(atmo_flns, band_fln, maxv=500e3, deltav=1e3, resample=None, smooth=None, wrange=None, verbose=False):
    """
    !!!NEEDS AN UPGRADE!!!

    This function calculates the Doppler boosting factor of BT-Settl spectra
    given a certain bandpass.

    atmo_flns (list): List of BT-Settl filenames. Can be h5, or .7 format.
    band_fln (str): Atmosphere filename.
    maxv (float): maximum value for the Doppler shift to be sampled in m/s
    deltav (float): Doppler shift sampling in m/s
    resample (float): If provided, will resample the spectrum at the value (in
        Armstroms).
    smooth (float): If provided, will convolve the spectrum with a gaussian kernel
        of sigma=smooth.
    wrange (list): One can provide the minimum and maximum value of wavelength
        to keep.
    verbose (bool): If true, will display the a plot of the fit for the Doppler boosting.
    """
    # Load the filter interpolator
    band_func = Filter.Load_filter(band_fln, nu=False)
    
    # Iterating through the atmosphere data to calculate the Doppler boosting
    boost = []
    atmo_flns.sort()
    for atmo_fln in atmo_flns:
        print( 'Doppler boosting for %s' %atmo_fln )
        if atmo_fln.find('.fits') != -1:
            import pyfits
            hdulist = pyfits.open(atmo_fln)
            hdu1 = hdulist[1]
            tbdata = hdu1.data
            f = tbdata.field(1)
            w = tbdata.field(0)
        else:
            w, f = Load_BTSettl_AGSS2009(atmo_fln, nu=False, wrange=wrange, resample=resample)
        if smooth is not None:
            f = scipy.ndimage.gaussian_filter1d(f, smooth)
        band = band_func(w)
        boost.append( Filter.Doppler_boosting_factor(f, band, w, verbose=verbose) )
    return numpy.array(boost)

def Load_BTSettl_AGSS2009(atmo_fln, nu=True):
    """
    Load BTSettl AGSS2009 atmosphere models.
    
    atmo_fln: BT-Settl.h5 filename (HDF format).
    nu (True): if True returns w, f in Hz, erg/s/cm^2/Hz
        if False: returns w, f in A, erg/s/cm^2/A
    """
    atmo = h5py.File(atmo_fln, 'r')
    f_wav = atmo['Spectrum']['flux'].value # in erg/s/cm^2/cm
    f_wav = f_wav*1e-8 # in erg/s/cm^2/A
    ## If the wavelength data exist we read them
    try:
        wav = atmo['Spectrum']['wl'].value # in angstrom
    ## If it doesn't exist then we reconstruct it
    except:
        try:
            ber = atmo['Spectrum']['cmtber'].value
            dis = atmo['Spectrum']['cmtdis'].value
            ind = dis.tolist().index(0)
            npt = np.round((ber[1:ind+1]-ber[:ind]) / dis[:ind])
            wav = [ np.linspace(ber[i], ber[i+1], npt[i], endpoint=False) for i in range(ind) ] + [ ber[ind:ind+1] ]
            wav = np.concatenate(wav)
        except:
            print( "Error constructing the wavelength grid from cmtber info for file {}".format(atmo_fln) )
            wav = np.arange(f_wav.size)
    f_wav = np.asarray(f_wav, dtype=float)
    wav = np.asarray(wav, dtype=float)
    ## We sort the atmosphere data by wavelength because they are not always in the right order. Sometimes there are duplicated spectral bins. We weed them out and this takes care of sorting the bins in ascending order too.
    wav, ind = np.unique(wav, return_index=True)
    f_wav = f_wav[ind]
    ## We convert to frequency and F_nu if requested
    if nu:
        wav = wav * 1e-10 # convert A to m
        f_wav = f_wav * 1e10 # convert erg/s/cm^2/A to erg/s/cm^2/m
        f_nu = f_wav * wav**2 / cts.c # convert F_lambda to F_nu using Bessell & Murphy 2012 (eq. A1)
        nu = cts.c / wav # convert wavelength to frequency
        ## Invert the order to make it ascending function of frequency        
        f_nu = f_nu[::-1]
        nu = nu[::-1]
        return nu, f_nu
    else:
        return wav, f_wav


##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##
## Contain functions to process BT-Settl.7 spectra
##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##

def Bandint_BTSettl7(atmo_flns, band_fln, output_fln, nu=True):
    """
    Integrate BTSettl7 atmosphere models (ascii format) through
    a given filter.
    
    atmo_flns: atmosphere model filenames (ascii format).
    band_fln: filter filename.
    output_fln: output filename.
    nu (True): Whether the integration should be performed in the frequency
        (STMAG system) or wavelength (ABMAG system) space.
    """
    # Load the filter interpolator
    band_func = Load_filter(band_fln, nu=nu)
    
    # Iterating through the atmosphere data to integrate through the filter
    atmo_flns.sort()
    with open(output_fln, 'w') as output:
        for atmo_fln in atmo_flns:
            print( 'Integrating %s' %atmo_fln )
            ind = atmo_fln.find('lte')
            temp = float(atmo_fln[ind+3:ind+6])*100
            logg = float(atmo_fln[ind+7:ind+10])
            if atmo_fln[ind+6] == '+':
                logg *= -1
            w, f = Load_BTSettl7(atmo_fln, nu=nu)
            flux = Filter.Band_integration(band_func, w, f, nu=nu)
            output.write( '%7.1f %4.2f %e\n' %(temp,logg,flux) )
    return

def Load_BTSettl7(atmo_fln, nu=True):
    """
    Load BTSettl7 atmosphere models.
    
    atmo_fln: BT-Settl.7 filename (ascii format).
    nu (True): if True returns w, f in Hz, erg/s/cm^2/Hz
        if False: returns w, f in A, erg/s/cm^2/A
    """
    ## Load the atmosphere data, first column is wavelength in A, second column is log10(flux), in ergs/s/cm**2/cm
    atmo = np.loadtxt(atmo_fln, unpack=True, usecols=(0,1), converters={0: lambda s: float(s.replace('D','E')), 1: lambda s: float(s.replace('D','E'))})
    f_wav = atmo[1] # in log10(erg/s/cm^2/cm)
    f_wav = 10**(f_wav-8) # in erg/s/cm^2/A
    wav = atmo[0] # in A
    ## We sort the atmosphere data by wavelength because they are not always in the right order. Sometimes there are duplicated spectral bins. We weed them out and this takes care of sorting the bins in ascending order too.
    wav, ind = np.unique(wav, return_index=True)
    f_wav = f_wav[ind]
    ## We convert to frequency and F_nu if requested
    if nu:
        wav = wav * 1e-10 # convert A to m
        f_wav = f_wav * 1e10 # convert erg/s/cm^2/A to erg/s/cm^2/m
        f_nu = f_wav * wav**2 / cts.c # convert F_lambda to F_nu using Bessell & Murphy 2012 (eq. A1)
        nu = cts.c / wav # convert wavelength to frequency
        ## Invert the order to make it ascending function of frequency        
        f_nu = f_nu[::-1]
        nu = nu[::-1]
        return nu, f_nu
    else:
        return wav, f_wav


##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##
## Contain functions to process BT-Settl.server.spec.9 spectra
##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##

def Trim_negative(fln, fln_out):
    """ Trim_negative(fln, fln_out)
    Process a .9 files have a certain number of mu values, the first
    half of which are negative and therefore useless. The data will be
    reformated as following:
        # mu1, mu2, ...,
        wavelenght F_nu(mu1) F_nu(mu2), ...,
    
    fln: file object or filename of the input data.
    fln_out: filename of the output data.
    
    >>> Trim_negative('lte.BT-Settl.server.spec.9', 'lte.BT-Settl.server.spec.9.trim')
    """
    if isinstance(fln, basestring):
        # Open the file if a string is passed
        print( '##### #####\nTrimming negative ray angles from file %s' %fln )
        f = open(fln)
    else:
        f = fln
    # Read the first line: number of ray directions (mu = cos(theta))
    ndirections = int(f.readline().strip())
    # Return to the beginning of the file
    f.seek(0)
    # Read all the data in the buffer
    data = f.read()
    f.close()
    # Split everything that is separated by one or many spaces
    data = data.split()
    # In principle, at each frequency there should be the frequency value + ndirections F_nu values.
    if len(data)%(ndirections+1) != 0:
        print( "There is a problem with the source file, ndata\%(ndirections+1) != 0." )
    # Creating the output file
    print( 'Saving the results in file %s' %fln_out )
    fout = open(fln_out, 'w')
    niter = len(data)/(ndirections+1)
    ntot = ndirections+1
    nhalf = ndirections/2+1
    # Write the new header
    print( 'Writing the header information' )
    fout.write( '# ' + ' '.join(data[nhalf:ntot]) + '\n' )
    print( 'Sorting the wavelengths' )
    wavelengths = numpy.array([ data[i*ntot] for i in xrange(1,niter) ], dtype=float)
    inds = wavelengths.argsort()+1
    print( 'Writing the data' )
    [ fout.write( data[i*ntot] + ' ' + ' '.join(data[i*ntot+nhalf:i*ntot+ndirections+1]) + '\n' ) for i in inds ]
    print( 'Done!' )
    fout.close()
    return

def Read_header(fln):
    """ Read_header(fln)
    Read the file header information (number of mu directions and mu
    values) and returns them.
    
    fln: file object or filename.
    
    >>> ndirections, mu = Read_header(fln)
    """
    if isinstance(fln, basestring):
        # Open the file if a string is passed
        f = open(fln)
    else:
        f = fln
    # Read the first line: number of ray directions (mu = cos(theta))
    ndirections = int(f.readline().strip())
    # There are a maximum of 10 data per line, so calculate how many lines hold the information for one wavelength.
    nlines = int(numpy.ceil(ndirections/10.))
    # Read the mu values
    mu = numpy.empty(0)
    for i in xrange(nlines):
        line = f.readline()
        mu = numpy.r_[ mu, numpy.array(line.strip().split(), dtype=float) ]
    return ndirections, mu


##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##
## General functions
##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##

def Fix_exponentials(fln):
    """
    Several BTSettl7 ascii files use the old Fortran exponential
    notation using D instead of E. This script fixes it, in place.
    """
    from subprocess import call
    print( "Fixing the exponentials for {}".format(fln) )
    bz2 = fln.endswith('.bz2')
    if bz2:
        call("bunzip2 {}".format(fln), shell=True)
    call("sed -i.bkp s/D/E/g {}".format(fln.rsplit('.bz2', 1)), shell=True)
    if bz2:
        call("bzip2 {}".format(fln), shell=True)
    return


##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##
## This is the autoexec
##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##
if __name__ == "__main__":
    if 0:
        print( "Batch processing of BT-Settl.server.spec.9 files" )
        flns = glob.glob("lte*.BT-Settl.server.spec.9")
        for fln in flns:
            Fix_exponentials(fln)
            Trim_negative(fln, fln+".trim")




