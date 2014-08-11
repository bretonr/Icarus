# Licensed under a 3-clause BSD style license - see LICENSE

__all__ = ["AtmoGrid", "AtmoGridPhot"]

from ..Utils.import_modules import *
from .. import Utils
from astropy.table import TableColumns, Column, MaskedColumn
from astropy.utils.metadata import MetaData
from astropy.utils import OrderedDict
from astropy.extern import six


######################## class Atmo_grid ########################
class AtmoGrid(Column):
    """
    Define the base atmosphere grid structure.

    AtmoGrid contains utilities to trim the grid, read/write to HDF5 format.

    Parameters
    ----------
    data : ndarray
        Grid of log(flux) values (e-base)
    name : str
        Keyword name of the atmosphere grid
    dtype : numpy.dtype compatible value
        Data type the flux grid
    shape : tuple or ()
        Dimensions of a single row element in the flux grid
    length : int or 0
        Number of row elements in the grid
    description : str or None
        Full description of the atmosphere grid
    unit : str or None
        Physical unit
    format : str or None or function or callable
        Format string for outputting column values.  This can be an
        "old-style" (``format % value``) or "new-style" (`str.format`)
        format specification string or a function or any callable object that
        accepts a single value and returns a string.
    meta : dict-like or None
        Meta-data associated with the atmosphere grid
    cols : OrderedDict-like, list of Columns, list of lists/tuples
        Full definition of the flux grid axes. This can be a list of entries
        ('colname', ndarray) with the ndarray corresponding to the axis values
        or a list of Columns containing this information.

    Examples
    --------
    A AtmoGrid can be created like this:

      Examples::

        logtemp = np.exp(np.arange(3000.,10001.,250.))
        logg = np.arange(2.0, 5.6, 0.5)
        mu = np.arange(0.,1.01,0.02)
        logflux = np.random.normal(size=(logtemp.size,logg.size,mu.size))
        atmo = AtmoGrid(data=logflux, cols=[('logtemp',logtemp), ('logg',logg), ('mu',mu)])

    Note that in principle the axis and data could be any format. However, we recommend using
    log(flux) and log(temperature) because the linear interpolation of such a grid would make
    more sense (say, from the blackbody $F \propto sigma T^4$).

    To read/save a file:

        atmo = AtmoGridPhot.ReadHDF5('vband.h5')
        atmo.WriteHDF5('vband_new.h5')
    """
    def __new__(cls, data=None, name=None, dtype=None, shape=(), length=0, description=None, unit=None, format=None, meta=None, cols=None):

        self = super(AtmoGrid, cls).__new__(cls, data=data, name=name, dtype=dtype, shape=shape, length=length, description=description, unit=unit, format=format, meta=meta)

        if cols is None:
            self.cols = TableColumns()
            for i in range(self.ndim):
                key = str(i)
                val = np.arange(self.shape[i])
                self.cols[key] = val
        else:
            if len(cols) != self.ndim:
                raise ValueError('cols must contain a number of elements equal to the dimension of the data grid.')
            else:
                try:
                    if isinstance(cols, (list,tuple)):
                        cols = OrderedDict(cols)
                    self.cols = TableColumns(cols)
                except:
                    raise ValueError('Cannot make a TableColumns out of the provided cols parameter.')
                shape = tuple(col.size for col in self.cols.itervalues())
                if self.shape != shape:
                    raise ValueError('The dimension of the data grid and the cols are not matching.')
        return self

    def __getitem__(self, item):
        if isinstance(item, six.string_types):
            return self.cols[item]
        else:
            return super(AtmoGrid, self).__getitem__(item)

    @property
    def colnames(self):
        return self.cols.keys()

    @classmethod
    def ReadHDF5(cls, fln):
        try:
            import h5py
        except ImportError:
            raise Exception("h5py is needed for ReadHDF5")
        f = h5py.File(fln, 'r')

        flux = f['flux'].value

        meta = {}
        for key_attrs, val_attrs in f.attrs.iteritems():
            meta[key_attrs] = val_attrs
        colnames = meta.pop('colnames')
        name = meta.pop('name')
        description = meta.pop('description')

        cols = []
        grp = f['cols']
        for col in colnames:
            dset = grp[col]
            cols.append( Column(data=dset.value, name=col, meta=dict(dset.attrs.iteritems())) )
        cols = TableColumns(cols)

        f.close()
        return cls(data=flux, name=name, description=description, meta=meta, cols=cols)

    def WriteHDF5(self, fln, overwrite=False):
        try:
            import h5py
        except ImportError:
            raise Exception("h5py is needed for WriteHDF5")

        if os.path.exists(fln):
            if overwrite:
                os.remove(fln)
            else:
                raise IOError("File exists: {}".format(fln))

        f = h5py.File(fln, 'w')

        f.create_dataset(name='flux', data=self.data)
        f.attrs['colnames'] = self.cols.keys()
        f.attrs['name'] = self.name
        f.attrs['description'] = self.description

        for key_attrs, val_attrs in self.meta.iteritems():
            f.attrs[key_attrs] = val_attrs

        grp = f.create_group('cols')
        for key, val in self.cols.iteritems():
            dset = grp.create_dataset(name=key, data=val)
            if hasattr(val, 'meta'):
                for key_attrs, val_attrs in val.meta.iteritems():
                    dset.attrs[key_attrs] = val_attrs
        f.close()

    def Getaxispos(self, colname, x):
        """
        Return the index and weight of the linear interpolation of the point
        along a given axis.

        Parameters
        ----------
        colname : str
            Name of the axis to interpolate from.
        x : float, ndarray
            Value to interpolate at.

        Examples
        ----------
          Examples::
            temp = Getaxispos('logtemp', np.log(3550.)
            logg = Getaxispos('logg', [4.11,4.13,4.02])

        """
        if isinstance(x, (list, tuple, numpy.ndarray)):
            return Utils.Series.Getaxispos_vector(self.cols[colname], x)
        else:
            return Utils.Series.Getaxispos_scalar(self.cols[colname], x)

    def Trim(self, colname, low=None, high=None):
        """
        """
        if low is None:
            low = self.cols[colname].min()
        if high is None:
            high = self.cols[colname].max()
        raise Exception('Needs to be completed!!!')


class AtmoGridPhot(AtmoGrid):
    """
    Define a subclass of AtmoGrid dedicated to Inter8_photometry

    This class is exactly like AtmoGrid, except for the fact that it is
    required to contain a seta of meta data describing the filter. For
    instance meta should contain:

    zp: float
        The zeropoint of the band for conversion from flux to mag
    ext: float
        The Aband/Av extinction ratio
    
    Also recommended would be:
    filter: str
        Description of the filter
    w0: float
        Central wavelength of the filter
    delta_w: float
        Width of the filter
    pivot: float
        Pivot wavelength
    units: str
        Description of the flux units
    magsys: str
        Magnitude system

    Examples
    --------
    A AtmoGrid can be created like this:

      Examples::

        logtemp = np.exp(np.arange(3000.,10001.,250.))
        logg = np.arange(2.0, 5.6, 0.5)
        mu = np.arange(0.,1.01,0.02)
        logflux = np.random.normal(size=(logtemp.size,logg.size,mu.size))
        atmo = AtmoGrid(data=logflux, cols=[('logtemp',logtemp), ('logg',logg), ('mu',mu)])

    Note that in principle the axis and data could be any format. However, we recommend using
    log(flux) and log(temperature) because the linear interpolation of such a grid would make
    more sense (say, from the blackbody $F \propto sigma T^4$).

    To read/save a file:

        atmo = AtmoGridPhot.ReadHDF5('vband.h5')
        atmo.WriteHDF5('vband_new.h5')
    """
    def __new__(cls, *args, **kwargs):
        self = super(AtmoGridPhot, cls).__new__(cls, *args, **kwargs)

        ## This class requires a certain number of keywords in the meta field
        if 'zp' not in self.meta:
            self.meta['zp'] = 0.0
        if 'ext' not in self.meta:
            self.meta['ext'] = 1.0

        return self

    def Get_flux(self, val_logtemp, val_logg, val_mu, val_area):
        """
        Return the flux interpolated from the atmosphere grid.
        
        Parameters
        ----------
        val_logtemp: log effective temperature
        val_logg: log surface gravity
        val_mu: cos(angle) of angle of emission
        val_area: area of the surface element
        
        Examples
        ----------
          Examples::
            flux = Get_flux(val_logtemp, val_logg, val_mu, val_area)
        """
        w1logtemp, jlogtemp = self.Getaxispos('logtemp', val_logtemp)
        w1logg, jlogg = self.Getaxispos('logg', val_logg)
        w1mu, jmu = self.Getaxispos('mu', val_mu)
        flux = Utils.Grid.Inter8_photometry(self.data, w1logtemp, w1logg, w1mu, jlogtemp, jlogg, jmu, val_area, val_mu)
        return flux

    def Get_flux_details(self, val_logtemp, val_logg, val_mu, val_area, val_v):
        """
        Returns the flux interpolated from the atmosphere grid.

        Parameters
        ----------
        val_logtemp: log of effective temperature
        val_logg: log of surface gravity
        val_mu: cos(angle) of angle of emission
        val_area: area of the surface element
        val_v: velocity of the surface element
        
        Examples
        ----------
          Examples::
            flux, Keff, temp = Get_flux_details(val_logtemp, val_logg, val_mu, val_area, val_v)
        """
        w1logtemp, jlogtemp = self.Getaxispos('logtemp', val_logtemp)
        w1logg, jlogg = self.Getaxispos('logg', val_logg)
        w1mu, jmu = self.Getaxispos('mu', val_mu)
        flux, Keff, temp = Utils.Grid.Inter8_photometry_details(self.data, w1logtemp, w1logg, w1mu, jlogtemp, jlogg, jmu, val_area, val_mu, val_v, val_logtemp)
        return flux, Keff, temp

    def Get_flux_Keff(self, val_temp, val_logg, val_mu, val_area, val_v):
        """
        Returns the flux interpolated from the atmosphere grid.

        Parameters
        ----------
        val_logtemp: log of effective temperature
        val_logg: log of surface gravity
        val_mu: cos(angle) of angle of emission
        val_area: area of the surface element
        val_v: velocity of the surface element
        
        Examples
        ----------
          Examples::
            flux, Keff = Get_flux_Keff(val_logtemp, val_logg, val_mu, val_area, val_v)
        """
        w1logtemp, jlogtemp = self.Getaxispos('logtemp', val_logtemp)
        w1logg, jlogg = self.Getaxispos('logg', val_logg)
        w1mu, jmu = self.Getaxispos('mu', val_mu)
        flux, Keff = Utils.Grid.Inter8_photometry_Keff(self.data, w1logtemp, w1logg, w1mu, jlogtemp, jlogg, jmu, val_area, val_mu, val_v)
        return flux, Keff

    def Get_flux_nosum(self, val_logtemp, val_logg, val_mu, val_area):
        """
        Returns the flux interpolated from the atmosphere grid.

        Parameters
        ----------
        val_logtemp: log of effective temperature
        val_logg: log of surface gravity
        val_mu: cos(angle) of angle of emission
        val_area: area of the surface element
        
        Examples
        ----------
          Examples::
            fluxes = Get_flux_nosum(val_logtemp, val_logg, val_mu, val_area)
        """
        w1logtemp, jlogtemp = self.Getaxispos('logtemp', val_logtemp)
        w1logg, jlogg = self.Getaxispos('logg', val_logg)
        w1mu, jmu = self.Getaxispos('mu', val_mu)
        flux = Utils.Grid.Inter8_photometry_nosum(self.data, w1logtemp, w1logg, w1mu, jlogtemp, jlogg, jmu, val_area, val_mu)
        return flux


class Atmo_grid:
    """
    This class handles the atmosphere grid.
    """
    def __init__(self, fln, wav, dwav, zp, ext=0.):
        """__init__
        """
        self.wav = wav
        self.dwav = dwav
        self.zp = zp
        self.ext = ext
        self.fln = fln
        self.Flux_init()

    def Flux_init(self):
        """Flux_init()
        Reads a band file and construct a grid.
        
        Calculates:
            temp: effective temperatures. temp.shape = (ntemp)
            logg: log of surface gravity. logg.shape = (nlogg)
            mu: cos(angle) of emission direction. mu.shape = (nmu)
            grid: the grid of specific intensities. grid.shape = (ntemp,nlogg,nmu)
            leff: ???
            h: ???
        
        >>> self.Flux_init()
        """
        f = open(self.fln,'r')
        lines = f.readlines()
        # We read the header line containing the number of temperatures (n_temp), logg (n_logg) and mu=cos(angle) (n_mu)
        n_temp, n_logg, n_mu = lines[1].split()[:3]
        n_temp = int(n_temp)
        n_logg = int(n_logg)
        n_mu = int(n_mu)
        # There should be 3 lines per grid point (temp,logg,mu): the info line and two flux lines
        # To that, we must subtract the comment line, the header line and two lines for the mu values
        if (n_temp*abs(n_logg)*3) != len(lines)-4:
            print 'It appears that the number of lines in the file is weird'
            return None
        # Read the mu values
        mu = numpy.array(lines[2].split()+lines[3].split(),dtype=float)
        # Read the info line for each grid point
        hdr = []
        grid = []
        for i in numpy.arange(4,len(lines),3):
            hdr.append(lines[i].split())
            grid.append(lines[i+1].split()+lines[i+2].split())
        hdr = numpy.array(hdr,dtype=float)
        grid = numpy.log(numpy.array(grid,dtype=float)/(C*100)*self.wav**2)
        hdr.shape = (n_temp,abs(n_logg),hdr.shape[1])
        grid.shape = (n_temp,abs(n_logg),n_mu)
        logtemp = numpy.log(hdr[:,0,0])
        logg = hdr[0,:,1]
        leff = hdr[0,0,2]
        #jl = hdr[:,:,3]
        h = hdr[:,:,4]
        #bl = hdr[:,:,5]
        #self.hdr = hdr
        self.grid = grid
        self.logtemp = logtemp
        self.logg = logg
        self.mu = mu
        self.leff = leff
        self.h = h
        return

    def Get_flux(self, val_temp, val_logg, val_mu, val_area):
        """
        Returns the flux interpolated from the atmosphere grid.
        val_temp: log of effective temperature
        val_logg: log of surface gravity
        val_mu: cos(angle) of angle of emission
        val_area: area of the surface element
        
        >>> self.Get_flux(val_temp, val_logg, val_mu, val_area)
        flux
        """
        grid = self.grid
        logtemp = self.logtemp
        logg = self.logg
        mu = self.mu
        w1temp, jtemp = self.Getaxispos(logtemp,val_temp)
        w1logg, jlogg = self.Getaxispos(logg,val_logg)
        w1mu, jmu = self.Getaxispos(mu,val_mu)
        flux = Utils.Grid.Inter8_photometry(grid, w1temp, w1logg, w1mu, jtemp, jlogg, jmu, val_area, val_mu)
        return flux

    def Get_flux_details(self, val_temp, val_logg, val_mu, val_area, val_v):
        """
        Returns the flux interpolated from the atmosphere grid.
        val_temp: log of effective temperature
        val_logg: log of surface gravity
        val_mu: cos(angle) of angle of emission
        val_area: area of the surface element
        val_v: velocity of the surface element
        
        >>> self.Get_flux_details(val_temp, val_logg, val_mu, val_area, val_v)
        flux, Keff, temp
        """
        grid = self.grid
        logtemp = self.logtemp
        logg = self.logg
        mu = self.mu
        w1temp, jtemp = self.Getaxispos(logtemp,val_temp)
        w1logg, jlogg = self.Getaxispos(logg,val_logg)
        w1mu, jmu = self.Getaxispos(mu,val_mu)
        flux, Keff, temp = Utils.Grid.Inter8_photometry_details(grid, w1temp, w1logg, w1mu, jtemp, jlogg, jmu, val_area, val_mu, val_v, val_temp)
        return flux, Keff, temp

    def Get_flux_Keff(self, val_temp, val_logg, val_mu, val_area, val_v):
        """
        Returns the flux interpolated from the atmosphere grid.
        val_temp: log of effective temperature
        val_logg: log of surface gravity
        val_mu: cos(angle) of angle of emission
        val_area: area of the surface element
        val_v: velocity of the surface element
        
        >>> self.Get_flux_Keff(val_temp, val_logg, val_mu, val_area, val_v)
        flux, Keff
        """
        grid = self.grid
        logtemp = self.logtemp
        logg = self.logg
        mu = self.mu
        w1temp, jtemp = self.Getaxispos(logtemp,val_temp)
        w1logg, jlogg = self.Getaxispos(logg,val_logg)
        w1mu, jmu = self.Getaxispos(mu,val_mu)
        flux, Keff = Utils.Grid.Inter8_photometry_Keff(grid, w1temp, w1logg, w1mu, jtemp, jlogg, jmu, val_area, val_mu, val_v)
        return flux, Keff

    def Get_flux_nosum(self, val_temp, val_logg, val_mu, val_area):
        """
        Returns the flux interpolated from the atmosphere grid.
        val_temp: log of effective temperature
        val_logg: log of surface gravity
        val_mu: cos(angle) of angle of emission
        val_area: area of the surface element
        
        >>> self.Get_flux_nosum(val_temp, val_logg, val_mu, val_area)
        flux
        """
        grid = self.grid
        logtemp = self.logtemp
        logg = self.logg
        mu = self.mu
        w1temp, jtemp = self.Getaxispos(logtemp,val_temp)
        w1logg, jlogg = self.Getaxispos(logg,val_logg)
        w1mu, jmu = self.Getaxispos(mu,val_mu)
        flux = Utils.Grid.Inter8_photometry_nosum(grid, w1temp, w1logg, w1mu, jtemp, jlogg, jmu, val_area, val_mu)
        return flux

    def Getaxispos(self, xx, x):
        """
        """
        if isinstance(x, (list, tuple, numpy.ndarray)):
            return Utils.Series.Getaxispos_vector(xx, x)
        else:
            return Utils.Series.Getaxispos_scalar(xx, x)

    def Getaxispos_old(self, xx, x):
        """
        OBSOLETE!
        """
        ascending = xx[-1] > xx[0]
        jl = 0
        ju = xx.size
        while (ju-jl) > 1:
            jm=(ju+jl)/2
            if ascending == (x > xx[jm]):
                jl=jm
            else:
                ju=jm
        j = min(max(jl,0),xx.size-1)
        if j == xx.size-1:
            j -= 1
            #print "Reaching the end..."
        w = (x-xx[j])/(xx[j+1]-xx[j])
        return w,j

    def Inter8_orig(self, val_temp, val_logg, val_mu):
        """
        Obsolete!!!
        """
        grid = self.grid
        logtemp = self.logtemp
        logg = self.logg
        mu = self.mu
        w1temp, jtemp = self.Getaxispos(logtemp,val_temp)
        w1logg, jlogg = self.Getaxispos(logg,val_logg)
        w1mu, jmu = self.Getaxispos(mu,val_mu)
        w0mu = 1.-w1mu
        w0temp = 1.-w1temp
        w0logg = 1.-w1logg
        fl = w0logg*(w0temp*(w0mu*grid[jtemp,jlogg,jmu] \
                            +w1mu*grid[jtemp,jlogg,jmu+1]) \
                    +w1temp*(w0mu*grid[jtemp+1,jlogg,jmu] \
                            +w1mu*grid[jtemp+1,jlogg,jmu+1])) \
            +w1logg*(w0temp*(w0mu*grid[jtemp,jlogg+1,jmu] \
                            +w1mu*grid[jtemp,jlogg+1,jmu+1]) \
                    +w1temp*(w0mu*grid[jtemp+1,jlogg+1,jmu] \
                            +w1mu*grid[jtemp+1,jlogg+1,jmu+1]))
        flux = numpy.exp(fl)*val_mu
        return flux

######################## class Atmo_grid ########################

