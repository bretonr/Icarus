# Licensed under a 3-clause BSD style license - see LICENSE

from .import_modules import *


##### ##### ##### ##### #####
##### This script allows to process BT-Settl.server.spec.9 files
##### ##### ##### ##### #####


def Preprocess(fln):
    """ Preprocess(fln)
    Preprocessing of the .9 files.
    
    At this point, one task is performed:
    1. Using sed, the exponent notation is changed from 'D' to 'E' to
    make it compatible with Python.
    """
    print( '##### #####\nPreprocessing file %s' %fln )
    system( "sed -i -e 's/D/E/g' %s" %fln )
    return

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
    if type(fln) is type(str()):
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
    if type(fln) is type(str()):
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



if __name__ == "__main__":
    print( "Batch processing of BT-Settl.server.spec.9 files" )
    flns = glob.glob("lte*.BT-Settl.server.spec.9")
    for fln in flns:
        Preprocess(fln)
        Trim_negative(fln, fln+".trim")




