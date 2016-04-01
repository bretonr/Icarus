# Licensed under a 3-clause BSD style license - see LICENSE

__all__ = ["AtmoGrid", "AtmoGridPhot", "AtmoGridDoppler", "AtmoGridSpec", "Vstack", "Atmo_grid"]

import os
import sys
from copy import deepcopy

from astropy.table import TableColumns, Column, MaskedColumn, Table
from astropy.utils.metadata import MetaData
from astropy.utils import OrderedDict
from astropy.extern import six

from ..Utils.import_modules import *
from .. import Utils

logger = logging.getLogger(__name__)


##-----------------------------------------------------------------------------
## class AtmoGrid
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
    dtype : np.dtype compatible value
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

        logtemp = np.log(np.arange(3000.,10001.,250.))
        logg = np.arange(2.0, 5.6, 0.5)
        mu = np.arange(0.,1.01,0.02)
        logflux = np.random.normal(size=(logtemp.size,logg.size,mu.size))
        atmo = AtmoGrid(data=logflux, cols=[('logtemp',logtemp), ('logg',logg), ('mu',mu)])

    To read/save a file:

        atmo = AtmoGridPhot.ReadHDF5('vband.h5')
        atmo.WriteHDF5('vband_new.h5')

    Notes
    --------------
    Note that in principle the axis and data could be any format. However, we recommend using
    log(flux) and log(temperature) because the linear interpolation of such a grid would make
    more sense (say, from the blackbody $F \propto sigma T^4$).

    """
    def __new__(cls, data=None, name=None, dtype=None, shape=(), length=0, description=None, unit=None, format=None, meta=None, cols=None):

        self = super(AtmoGrid, cls).__new__(cls, data=data, name=name, dtype=dtype, shape=shape, length=length, description=description, unit=unit, format=format, meta=meta)

        if cols is None:
            self.cols = TableColumns([ Column(name=str(i), data=np.arange(self.shape[i], dtype=float)) for i in range(self.ndim) ])
        else:
            if len(cols) != self.ndim:
                raise ValueError('cols must contain a number of elements equal to the dimension of the data grid.')
            else:
                if isinstance(cols, TableColumns):
                    self.cols = cols
                else:
                    try:
                        self.cols = TableColumns([ Column(name=col[0], data=col[1]) if isinstance(col, (list,tuple)) else col for col in cols ])
                    except:
                        raise ValueError('Cannot make a TableColumns out of the provided cols parameter.')
            shape = tuple(col.size for col in self.cols.itervalues())
            if self.shape != shape:
                raise ValueError('The dimension of the data grid and the cols are not matching.')
        return self

    def __copy__(self):
        return self.copy(copy_data=False)

    def __deepcopy__(self):
        return self.copy(copy_data=True)

    def __getitem__(self, item):
        if isinstance(item, six.string_types):
            if item not in self.colnames:
                if 'log'+item in self.colnames:
                    return np.exp(self.cols['log'+item])
                elif 'log'+item[2:] in self.colnames:
                    return 10**(self.cols['log'+item[2:]])
                elif 'log10'+item in self.colnames:
                    return 10**(self.cols['log10'+item])
                else:
                    raise Exception('The provided column name is cannot be found.')
            else:
                return self.cols[item]
        else:
            #return super(AtmoGrid, self).__getitem__(item)
            return self.view(np.ndarray)[item]

    @property
    def colnames(self):
        return self.cols.keys()

    def copy(self, order='C', data=None, copy_data=True):
        """
        Copy of the instance. If ``data`` is supplied
        then a view (reference) of ``data`` is used, and ``copy_data`` is ignored.
        """
        if data is None:
            data = self.view(np.ndarray)
            if copy_data:
                data = data.copy(order)

        return self.__class__(name=self.name, data=data, unit=self.unit, format=self.format, description=self.description, meta=deepcopy(self.meta), cols=self.cols)

    def Fill_nan_old(self, axis=0, method='spline', bounds_error=False, fill_value=np.nan, k=1, s=1):
        """
        Fill the empty grid cells (marked as np.nan) with interpolated values
        along a given axis (i.e. interpolation is done in 1D).

        Parameters
        ----------
        axis : interpolate
            Axis along which the interpolation should be performed.
        method : str
            Interpolation method to use. Possible choices are 'spline' and
            'interp1d'.
            'spline' allows for the use of optional keywords k (the order) and
            s (the smoothing parameter). See scipy.interpolate.splrep.
            'interp1d' allows for the use of optional keywords bounds_error and
            fill_value. See scipy.interpolate.interp1d.
        bounds_error : bool
            Whether to raise an error when attempting to extrapolate out of
            bounds. Only works with 'interp1d'.
        fill_value : float
            Value to use when bounds_error is False. Only works with 'interp1d'.
        k : int
            Order of the spline to use. We recommend 1. Only works with
            'spline'.
        s : int
            Smoothing parameter for the spline. We recommend 0 (exact
            interpolation), or 1. Only works with 'spline'.

        Examples
        ----------
          Examples::
            atmo.Fill_nan(axis=0, method='interp1d', bounds_error=False, fill_value=np.nan)
        
        This would fill in the value that are not out of bound with a linear fit. Values
        out of bound would be np.nan.

            atmo.Fill_nan(axis=0, method='spline', k=1, s=0)

        This would produce exactly the same interpolation as above, except that values
        out of bound would be extrapolated.

        Notes
        ----------
        From our experience, it is recommended to first fill the values within
        the bounds using 'interp1d' with bounds_error=False and fill_value=np.nan,
        and then use 'spline' with k=1 and s=1 in order to extrapolate outside
        the bounds. To interpolate within the bounds, the temperature axis
        (i.e. 0) is generally best and more smooth, whereas the logg axis (i.e. 1)
        works better to extrapolate outside.


          Examples::
            atmo.Fill_nan(axis=0, method='interp1d', bounds_error=False, fill_value=np.nan)
            atmo.Fill_nan(axis=1, method='spline', k=1, s=1)
        """
        if method not in ['interp1d','spline']:
            raise Exception('Wrong method input! Must be either interp1d, spline or grid.')
        ndim = list(self.shape)
        ndim.pop(axis)
        inds_tmp = np.indices(ndim)
        inds = [ind.flatten() for ind in inds_tmp]
        niter = len(inds[0])
        inds.insert(axis, [slice(None)]*niter)
        print(inds)
        for ind in zip(*inds):
            col = self.__getitem__(ind)
            inds_good = np.isfinite(col)
            inds_bad = ~inds_good
            if np.any(inds_bad):
                if method == 'interp1d':
                    interpolator = scipy.interpolate.interp1d(self.cols[axis][inds_good], col[inds_good], assume_sorted=True, bounds_error=bounds_error, fill_value=fill_value)
                    col[inds_bad] = interpolator(self.cols[axis][inds_bad])
                elif method == 'spline':
                    tck = scipy.interpolate.splrep(self.cols[axis][inds_good], col[inds_good], k=k, s=s)
                    col[inds_bad] = scipy.interpolate.splev(self.cols[axis][inds_bad], tck)

    def Fill_nan(self, axis=0, inds_fill=None, method='spline', bounds_error=False, fill_value=np.nan, k=1, s=0, extrapolate=True):
        """
        Fill the empty grid cells (marked as np.nan) with interpolated values
        along a given axis (i.e. interpolation is done in 1D).

        Parameters
        ----------
        axis : interpolate
            Axis along which the interpolation should be performed.
        inds_fill : tuple(ndarray)
            Tuple/list containing the list of pixels to interpolate for.
        method : str
            Interpolation method to use. Possible choices are 'spline', 
            'interp1d' and 'pchip'.
            'spline' allows for the use of optional keywords k (the order) and
            s (the smoothing parameter). See scipy.interpolate.splrep.
            'interp1d' allows for the use of optional keywords bounds_error and
            fill_value. See scipy.interpolate.interp1d.
            'pchip' allows to interpolate out of bound, or set NaNs.
        bounds_error : bool
            Whether to raise an error when attempting to extrapolate out of
            bounds.
            Only works with 'interp1d'.
        fill_value : float
            Value to use when bounds_error is False.
            Only works with 'interp1d'.
        k : int
            Order of the spline to use. We recommend 1.
            Only works with 'spline'.
        s : int
            Smoothing parameter for the spline. We recommend 0 (exact
            interpolation).
            Only works with 'spline'.
        extrapolate : bool
            Whether to extrapolate out of bound or set NaNs.
            Only works with 'pchip'.

        Examples
        ----------
          Examples::
            atmo.Fill_nan(axis=0, method='interp1d', bounds_error=False, fill_value=np.nan)
        
        This would fill in the value that are not out of bound with a linear fit. Values
        out of bound would be np.nan.

            atmo.Fill_nan(axis=0, method='spline', k=1, s=0)

        This would produce exactly the same interpolation as above, except that values
        out of bound would be extrapolated.

        Notes
        ----------
        From our experience, it is recommended to first fill the values within
        the bounds using 'interp1d' with bounds_error=False and fill_value=np.nan,
        and then use 'spline' with k=1 and s=1 in order to extrapolate outside
        the bounds. To interpolate within the bounds, the temperature axis
        (i.e. 0) is generally best and more smooth, whereas the logg axis (i.e. 1)
        works better to extrapolate outside.


          Examples::
            atmo.Fill_nan(axis=0, method='interp1d', bounds_error=False, fill_value=np.nan)
            atmo.Fill_nan(axis=1, method='spline', k=1, s=1)
        """
        if method not in ['interp1d','spline','pchip']:
            raise Exception('Wrong method input! Must be either interp1d, spline or grid.')
        if inds_fill is None:
            inds_fill = np.isnan(self.data).nonzero()
        else:
            assert len(inds_fill) == self.ndim, "The shape must be (ndim, nfill)."

        vals_fill = []
        for inds_fill_ in zip(*inds_fill):
            #print(inds_fill_)
            inds = list(inds_fill_)
            inds[axis] = slice(None)
            inds = tuple(inds)
            y = self.data[inds]
            x = self.cols[axis]
            x_interp = x[inds_fill_[axis]]
            #print(x)
            #print(y)
            #print(x_interp)
            inds_bad = np.isnan(y)
            inds_bad[x_interp] = True
            inds_good = ~inds_bad
            #print(inds_bad)
            #print(inds_good)
            if np.any(inds_good):
                #print(x[inds_good])
                #print(y[inds_good])
                #print(x_interp)
                if method == 'interp1d':
                    interpolator = scipy.interpolate.interp1d(x[inds_good], y[inds_good], assume_sorted=True, bounds_error=bounds_error, fill_value=fill_value)
                    y_interp = interpolator(x_interp)
                elif method == 'spline':
                    tck = scipy.interpolate.splrep(x[inds_good], y[inds_good], k=k, s=s)
                    y_interp = scipy.interpolate.splev(x_interp, tck)
                elif method == 'pchip':
                    interpolator = scipy.interpolate.PchipInterpolator(x[inds_good], y[inds_good], axis=0, extrapolate=extrapolate)
                    y_interp = interpolator(x_interp)
                #print(y_interp)
                vals_fill.append(y_interp)
            else:
                #print('y_interp -> nan')
                vals_fill.append(np.nan)
            #print('x_interp', x_interp)
            #print('y_interp', y_interp)
        self.data[inds_fill] = vals_fill

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
        if isinstance(x, (list, tuple, np.ndarray)):
            return Utils.Series.Getaxispos_vector(self.cols[colname], x)
        else:
            return Utils.Series.Getaxispos_scalar(self.cols[colname], x)

    @property
    def IsFinite(self):
        return np.isfinite(self.data).astype(int)

    def Pprint(self, slices):
        """
        Print a 2-dimensional slice of the atmosphere grid for visualisation.

        Parameters
        ----------
        slices : list
            List of sliceable elements to extract the 2-dim slice to display.

        Examples
        ----------
          Examples::
            # Display the equivalent of atmo[:,:,4]
            atmo.Pprint([None,None,4])
            # Same as above but using fancier slice objects
            atmo.Pprint([slice(None),slice(None),4])
            # Display the equivalent of atmo[3:9,3,:]
            atmo.Pprint([slice(3,9),3,None])
        """
        slices = list(slices)
        labels = []
        for i,s in enumerate(slices):
            if s is None:
                s = slice(None)
                slices[i] = s
            if isinstance(s, (int,slice)):
                tmp_label = self.cols[i][s]
                if self.colnames[i] == 'logtemp':
                    tmp_label = np.exp(tmp_label)
                if tmp_label.size > 1:
                    labels.append(tmp_label)
            else:
                raise Exception("The element {} is not a slice or integer or cannot be converted to a sliceable entity.".format(s))
        if len(labels) != 2:
            raise Exception("The slices should generate a 2 dimensional array. Verify your input slices.")
        t = Table(data=self.__getitem__(slices), names=labels[1].astype(str), copy=True)
        t.add_column(Column(data=labels[0]), index=0)
        t.pprint()

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

    def SubGrid(self, *args):
        """
        Return a sub-grid of the atmosphere grid.

        Parameters
        ----------
        slices : slice
            Slice/sliceable object for each dimension of the atmosphere grid.

        Examples
        ----------
          Examples::
            This would extract atmo[:,1:4,:]
            new_atmo = atmo.SubGrid(slice(None),slice(1,4),slice(None))
        """
        assert len(args) == self.ndim, "The number of slices must match the dimension of the atmosphere grid."
        slices = []
        for s in args:
            if isinstance(s,int):
                slices.append(slice(s,s+1))
            else:
                slices.append(s)
        data = self.data[slices]
        cols = []
        for c,s in zip(self.cols,slices):
            cols.append( (c, np.atleast_1d(self.cols[c][s])) )
        return self.__class__(name=self.name, data=data, unit=self.unit, format=self.format, description=self.description, meta=self.meta, cols=cols)

    def Trim(self, colname, low=None, high=None):
        """
        Return a copy of the atmosphere grid whose 'colname' axis has been
        trimmed at the 'low' and 'high' values: low <= colvalues <= high.

        Parameters
        ----------
        colname : str
            Name of the column to trim the grid on.
        low : float
            Lowest value to cut from. If None, will use the minimum value.
        high: float
            Highest value to cut from. If None, will use the maximum value.

        Examples
        ----------
          Examples::
          The following would trim along the temperature axis and keep values
          between 4000 and 6000, inclusively.
            new_atmo = atmo.Trim('logtemp', low=np.log(4000.), high=np.log(6000.))
        """
        if colname not in self.colnames:
            raise Exception("The provided column name is not valid.")
        colind = self.colnames.index(colname)
        cols = self.cols.copy()
        if low is None:
            low = cols[colname].min()
        if high is None:
            high = cols[colname].max()
        inds = [slice(None)]*self.ndim
        inds[colind] = np.logical_and(self.cols[colname] >= low, self.cols[colname] <= high)
        cols[colname] = Column(data=cols[colname][inds[colind]], name=colname)
        data = self.data[inds].copy()
        meta = deepcopy(self.meta)
        return self.__class__(name=self.name, data=data, unit=self.unit, format=self.format, description=self.description, meta=meta, cols=cols)

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


##-----------------------------------------------------------------------------
## class AtmoGridDoppler
class AtmoGridDoppler(AtmoGrid):
    """
    Define a subclass of AtmoGrid dedicated to storing the Doppler boosting
    factor.

    This class is exactly like AtmoGrid.

    Examples
    --------
    A AtmoGridDoppler can be created like this:

      Examples::

        logtemp = np.log(np.arange(3000.,10001.,250.))
        logg = np.arange(2.0, 5.6, 0.5)
        mu = np.arange(0.,1.01,0.02)
        boost = np.random.normal(size=(logtemp.size,logg.size,mu.size))
        atmo = AtmoGridDoppler(data=boost, cols=[('logtemp',logtemp), ('logg',logg), ('mu',mu)])

    Note that in principle the axis and data could be any format. However, we recommend using
    log(flux) and log(temperature) because the linear interpolation of such a grid would make
    more sense (say, from the blackbody $F \propto sigma T^4$).

    To read/save a file:

        atmo = AtmoGridDoppler.ReadHDF5('vband.h5')
        atmo.WriteHDF5('vband_new.h5')
    """
    def __new__(cls, *args, **kwargs):
        self = super(AtmoGridDoppler, cls).__new__(cls, *args, **kwargs)
        return self

    def Get_boost(self, val_logtemp, val_logg, val_mu):
        """
        Return the interpolated Doppler boosting factor
        
        Parameters
        ----------
        val_logtemp: log effective temperature
        val_logg: log surface gravity
        val_mu: cos(angle) of angle of emission
        
        Examples
        ----------
          Examples::
            boost = Get_boost(val_logtemp, val_logg, val_mu)
        """
        w1logtemp, jlogtemp = self.Getaxispos('logtemp', val_logtemp)
        w1logg, jlogg = self.Getaxispos('logg', val_logg)
        w1mu, jmu = self.Getaxispos('mu', val_mu)
        boost = Utils.Grid.Interp_3Dgrid(self.data, w1logtemp, w1logg, w1mu, jlogtemp, jlogg, jmu)
        return boost


##-----------------------------------------------------------------------------
## class AtmoGridPhot
class AtmoGridPhot(AtmoGrid):
    """
    Define a subclass of AtmoGrid dedicated to Photometry

    This class is exactly like AtmoGrid, except for the fact that it is
    required to contain a set of meta data describing the filter. For
    instance meta should contain:

    Parameters
    ----------
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
    A AtmoGridPhot can be created like this:

      Examples::

        logtemp = np.log(np.arange(3000.,10001.,250.))
        logg = np.arange(2.0, 5.6, 0.5)
        mu = np.arange(0.,1.01,0.02)
        logflux = np.random.normal(size=(logtemp.size,logg.size,mu.size))
        atmo = AtmoGridPhot(data=logflux, cols=[('logtemp',logtemp), ('logg',logg), ('mu',mu)])

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

    def Get_flux(self, val_logtemp, val_logg, val_mu, val_area, **kwargs):
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
        flux = Utils.Grid.Interp_photometry(self.data, w1logtemp, w1logg, w1mu, jlogtemp, jlogg, jmu, val_area, val_mu)
        return flux

    def Get_flux_details(self, val_logtemp, val_logg, val_mu, val_area, val_v, **kwargs):
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
            flux, Keff, vsini, temp = Get_flux_details(val_logtemp, val_logg, val_mu, val_area, val_v)
        """
        w1logtemp, jlogtemp = self.Getaxispos('logtemp', val_logtemp)
        w1logg, jlogg = self.Getaxispos('logg', val_logg)
        w1mu, jmu = self.Getaxispos('mu', val_mu)
        flux, Keff, vsini, temp = Utils.Grid.Interp_photometry_details(self.data, w1logtemp, w1logg, w1mu, jlogtemp, jlogg, jmu, val_area, val_mu, val_v, val_logtemp)
        return flux, Keff, vsini, temp

    def Get_flux_doppler(self, val_logtemp, val_logg, val_mu, val_area, val_vel, atmo_doppler, **kwargs):
        """
        Return the flux interpolated from the atmosphere grid.
        Each surface element is multiplied by the appropriate Doppler boosting
        factor.
        
        Parameters
        ----------
        val_logtemp: log effective temperature
        val_logg: log surface gravity
        val_mu: cos(angle) of angle of emission
        val_area: area of the surface element
        val_vel: velocity of the surface element (units of c)
        atmo_doppler: AtmoGridDoppler instance containing a grid of Doppler
            boosting factors. Must be the same dimensions as the atmosphere grid.
        
        Examples
        ----------
          Examples::
            flux = Get_flux_doppler(val_logtemp, val_logg, val_mu, val_area, val_vel, atmo_doppler)
        """
        w1logtemp, jlogtemp = self.Getaxispos('logtemp', val_logtemp)
        w1logg, jlogg = self.Getaxispos('logg', val_logg)
        w1mu, jmu = self.Getaxispos('mu', val_mu)
        flux = Utils.Grid.Interp_photometry_doppler(self.data, w1logtemp, w1logg, w1mu, jlogtemp, jlogg, jmu, val_area, val_mu, val_vel, atmo_doppler.data)
        return flux

    def Get_flux_doppler_nosum(self, val_logtemp, val_logg, val_mu, val_area, val_vel, atmo_doppler, **kwargs):
        """
        Return the flux interpolated from the atmosphere grid.
        Each surface element is multiplied by the appropriate Doppler boosting
        factor.
        
        Parameters
        ----------
        val_logtemp: log effective temperature
        val_logg: log surface gravity
        val_mu: cos(angle) of angle of emission
        val_area: area of the surface element
        val_vel: velocity of the surface element (units of c)
        atmo_doppler: AtmoGridDoppler instance containing a grid of Doppler
            boosting factors. Must be the same dimensions as the atmosphere grid.

        
        Examples
        ----------
          Examples::
            flux = Get_flux_doppler_nosum(val_logtemp, val_logg, val_mu, val_area, val_vel, atmo_doppler)
        """
        w1logtemp, jlogtemp = self.Getaxispos('logtemp', val_logtemp)
        w1logg, jlogg = self.Getaxispos('logg', val_logg)
        w1mu, jmu = self.Getaxispos('mu', val_mu)
        flux = Utils.Grid.Interp_photometry_doppler_nosum(self.data, w1logtemp, w1logg, w1mu, jlogtemp, jlogg, jmu, val_area, val_mu, val_vel, atmo_doppler.data)
        return flux

    def Get_flux_Keff(self, val_logtemp, val_logg, val_mu, val_area, val_v, **kwargs):
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
        flux, Keff = Utils.Grid.Interp_photometry_Keff(self.data, w1logtemp, w1logg, w1mu, jlogtemp, jlogg, jmu, val_area, val_mu, val_v)
        return flux, Keff

    def Get_flux_nosum(self, val_logtemp, val_logg, val_mu, val_area, **kwargs):
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
        flux = Utils.Grid.Interp_photometry_nosum(self.data, w1logtemp, w1logg, w1mu, jlogtemp, jlogg, jmu, val_area, val_mu)
        return flux


##-----------------------------------------------------------------------------
## class AtmoGridSpec
class AtmoGridSpec(AtmoGrid):
    """
    Define a subclass of AtmoGrid dedicated to Spectroscopy

    This class is exactly like AtmoGrid, except for the fact that it is
    required to contain a set of meta data describing the spectrum. For
    instance meta should contain:

    In principle, nothing prevents the wavelength to have any format. However,
    it is VERY recommended to use a linear spacing in log(wav), i.e. constant
    in velocity offset. The Get_flux method assumes that this is the case.

    Parameters
    ----------
    zp : float
        The zeropoint of the band for conversion from flux to mag
    delta_v : float
        Spacing between wavelength points in v/c units. The assumption is that
        it is constant throughout the array (in log(wav) space). If not
        provided, will use (wav[1]-wav[0])/wav[0].

    Also recommended would be:
    units: str
        Description of the flux units
    magsys: str
        Magnitude system

    Examples
    --------
    A AtmoGridSpec can be created like this:

      Examples::

        logtemp = np.log(np.arange(3000.,10001.,250.))
        logg = np.arange(2.0, 5.6, 0.5)
        mu = np.arange(0.,1.01,0.02)
        wav = np.arange(3000.,10000.5,1.)
        logflux = np.random.normal(size=(logtemp.size,logg.size,mu.size,wav.size))
        atmo = AtmoGridSpec(data=logflux, cols=[('logtemp',logtemp), ('logg',logg), ('mu',mu), ('wav',wav)])

    Note that in principle the axis and data could be any format. However, we recommend using
    log(flux) and log(temperature) because the linear interpolation of such a grid would make
    more sense (say, from the blackbody $F \propto sigma T^4$).

    To read/save a file:

        atmo = AtmoGridPhot.ReadHDF5('spectrum.h5')
        atmo.WriteHDF5('spectrum_new.h5')
    """
    def __new__(cls, *args, **kwargs):
        self = super(AtmoGridSpec, cls).__new__(cls, *args, **kwargs)

        ## This class requires a certain number of keywords in the meta field
        if 'zp' not in self.meta:
            self.meta['zp'] = 0.0
        if 'delta_v' not in self.meta:
            self.meta['delta_v'] = (self.cols['wav'][1]-self.cols['wav'][0]) / self.cols['wav'][0]

        return self

    def Get_flux_doppler(self, val_logtemp, val_logg, val_mu, val_area, val_vel, **kwargs):
        """
        Return the spectrum interpolated from the atmosphere grid.

        The returned spectrum remains on the same wavelength axis as the grid's
        wavelength axis.

        Parameters
        ----------
        val_logtemp: log effective temperature
        val_logg: log surface gravity
        val_mu: cos(angle) of angle of emission
        val_area: area of the surface element
        val_vel: velocity in v/c units.

        Examples
        ----------
          Examples::
            spectrum = Get_flux(val_logtemp, val_logg, val_mu, val_area, val_wav)
        """
        w1logtemp, jlogtemp = self.Getaxispos('logtemp', val_logtemp)
        w1logg, jlogg = self.Getaxispos('logg', val_logg)
        w1mu, jmu = self.Getaxispos('mu', val_mu)
        #w1wav, jwav = self.Getaxispos('wav', val_vel/self.meta['delta_v'])
        w1wav, jwav = np.modf(val_vel/self.meta['delta_v'])
        jwav = jwav.astype(int)
        logger.log(5, '-'*20)
        logger.log(5, 'w1logtemp')
        logger.log(5, w1logtemp)
        logger.log(5, '-'*20)
        logger.log(5, 'jlogtemp')
        logger.log(5, jlogtemp)
        logger.log(5, '-'*20)
        logger.log(5, 'w1logg')
        logger.log(5, w1logg)
        logger.log(5, '-'*20)
        logger.log(5, 'jlogg')
        logger.log(5, jlogg)
        logger.log(5, '-'*20)
        logger.log(5, 'w1mu')
        logger.log(5, w1mu)
        logger.log(5, '-'*20)
        logger.log(5, 'jmu')
        logger.log(5, jmu)
        logger.log(5, '-'*20)
        logger.log(5, 'w1wav')
        logger.log(5, w1wav)
        logger.log(5, '-'*20)
        logger.log(5, 'jwav')
        logger.log(5, jwav)
        logger.log(5, '-'*20)

        spectrum = Utils.Grid.Interp_doppler(self.data, w1logtemp, w1logg, w1mu, w1wav, jlogtemp, jlogg, jmu, jwav, val_area, val_mu)

        return spectrum

    @classmethod
    def ReadHDF5(cls, flns, verbose=True):
        if isinstance(flns, str):
            return super(AtmoGridSpec, cls).ReadHDF5(flns)

        try:
            import h5py
        except ImportError:
            raise Exception("h5py is needed for ReadHDF5")

        atmo = []
        logtemp = []
        logg = []
        for fln in flns:
            if verbose:
                sys.stdout.write('Loading {}\r'.format(fln.split('/')[-1]))
                sys.stdout.flush()
            atmo.append( cls.ReadHDF5(fln) )
            logtemp.append( atmo[-1]['logtemp'].data )
            logg.append( atmo[-1]['logg'].data )

        logtemp = np.unique(logtemp)
        logg = np.unique(logg)

        shape = [logtemp.size, logg.size, atmo[-1]['mu'].size, atmo[-1]['wav'].size]
        cols = [
            Column(data=logtemp, name='logtemp', meta=atmo[-1].cols['logtemp'].meta),
            Column(data=logg, name='logg', meta=atmo[-1].cols['logg'].meta),
            atmo[-1].cols['mu'],
            atmo[-1].cols['wav']
            ]
        meta = atmo[-1].meta
        description = atmo[-1].description
        name = atmo[-1].name

        if verbose:
            print("")
            print("Combining the grids")
        flux = np.full(shape, np.nan)
        for i,logtemp_ in enumerate(logtemp):
            for j,logg_ in enumerate(logg):
                for atmo_ in atmo:
                    if (logtemp_ in atmo_['logtemp']) and (logg_ in atmo_['logg']):
                        flux[i,j] = atmo_.data

        return cls(data=flux, name=name, description=description, meta=meta, cols=cols)


##-----------------------------------------------------------------------------
## Vstack function
def Vstack(grids, verbose=False):
    """
    Merges multiple atmosphere grids into a single grid.
    """
    ## Asserting that all grids have the same number of dimensions
    ndim = grids[0].ndim
    colnames = grids[0].colnames
    meta = grids[0].meta
    for grid in grids[1:]:
        if grid.ndim != ndim:
            raise Exception("Grids must all have the number of dimensions")
        if grid.colnames != colnames:
            raise Exception("Grids must all have the same dimension quantities")
        if grid.meta != meta:
            print("Caution: all the meta data are not the same.")

    ## Working out the columns of the new grid
    cols = []
    shape = []
    for i in range(ndim):
        vals = np.unique( np.hstack([g.cols[i] for g in grids]) )
        cols.append( (colnames[i], vals) )
        shape.append( vals.size )

    if verbose:
        print("The new grid will have a shape {}".format(shape))

    data = np.full(shape, np.nan, dtype=float)
    for i,grid in enumerate(grids):
        if verbose:
            print("Processing grid {}".format(i))
        inds = [ (grid.cols[j].data[:,None] == cols[j][1]).nonzero()[1] for j in range(ndim) ]
        inds = np.meshgrid(*inds, indexing='ij')
        data[inds] = grid

    new_grid = grids[0].__class__(data=data, cols=cols, meta=meta)
    return new_grid

def Quick_vstack(grids):
    """
    """


##-----------------------------------------------------------------------------
## class Atmo_Grid
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
        mu = np.array(lines[2].split()+lines[3].split(),dtype=float)
        # Read the info line for each grid point
        hdr = []
        grid = []
        for i in np.arange(4,len(lines),3):
            hdr.append(lines[i].split())
            grid.append(lines[i+1].split()+lines[i+2].split())
        hdr = np.array(hdr,dtype=float)
        grid = np.log(np.array(grid,dtype=float)/(cts.c*100)*self.wav**2)
        hdr.shape = (n_temp,abs(n_logg),hdr.shape[1])
        grid.shape = (n_temp,abs(n_logg),n_mu)
        logtemp = np.log(hdr[:,0,0])
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

    def Get_flux(self, val_logtemp, val_logg, val_mu, val_area, **kwargs):
        """
        Returns the flux interpolated from the atmosphere grid.
        val_logtemp: log of effective temperature
        val_logg: log of surface gravity
        val_mu: cos(angle) of angle of emission
        val_area: area of the surface element
        
        >>> self.Get_flux(val_logtemp, val_logg, val_mu, val_area)
        flux
        """
        grid = self.grid
        logtemp = self.logtemp
        logg = self.logg
        mu = self.mu
        w1temp, jtemp = self.Getaxispos(logtemp,val_logtemp)
        w1logg, jlogg = self.Getaxispos(logg,val_logg)
        w1mu, jmu = self.Getaxispos(mu,val_mu)
        flux = Utils.Grid.Interp_photometry(grid, w1temp, w1logg, w1mu, jtemp, jlogg, jmu, val_area, val_mu)
        return flux

    def Get_flux_details(self, val_logtemp, val_logg, val_mu, val_area, val_v, **kwargs):
        """
        Returns the flux interpolated from the atmosphere grid.
        val_logtemp: log of effective temperature
        val_logg: log of surface gravity
        val_mu: cos(angle) of angle of emission
        val_area: area of the surface element
        val_v: velocity of the surface element
        
        >>> self.Get_flux_details(val_logtemp, val_logg, val_mu, val_area, val_v)
        flux, Keff, vsini, temp
        """
        grid = self.grid
        logtemp = self.logtemp
        logg = self.logg
        mu = self.mu
        w1temp, jtemp = self.Getaxispos(logtemp,val_logtemp)
        w1logg, jlogg = self.Getaxispos(logg,val_logg)
        w1mu, jmu = self.Getaxispos(mu,val_mu)
        flux, Keff, vsini, temp = Utils.Grid.Interp_photometry_details(grid, w1temp, w1logg, w1mu, jtemp, jlogg, jmu, val_area, val_mu, val_v, val_logtemp)
        return flux, Keff, vsini, temp

    def Get_flux_Keff(self, val_logtemp, val_logg, val_mu, val_area, val_v, **kwargs):
        """
        Returns the flux interpolated from the atmosphere grid.
        val_logtemp: log of effective temperature
        val_logg: log of surface gravity
        val_mu: cos(angle) of angle of emission
        val_area: area of the surface element
        val_v: velocity of the surface element
        
        >>> self.Get_flux_Keff(val_logtemp, val_logg, val_mu, val_area, val_v)
        flux, Keff
        """
        grid = self.grid
        logtemp = self.logtemp
        logg = self.logg
        mu = self.mu
        w1temp, jtemp = self.Getaxispos(logtemp,val_logtemp)
        w1logg, jlogg = self.Getaxispos(logg,val_logg)
        w1mu, jmu = self.Getaxispos(mu,val_mu)
        flux, Keff = Utils.Grid.Interp_photometry_Keff(grid, w1temp, w1logg, w1mu, jtemp, jlogg, jmu, val_area, val_mu, val_v)
        return flux, Keff

    def Get_flux_nosum(self, val_logtemp, val_logg, val_mu, val_area, **kwargs):
        """
        Returns the flux interpolated from the atmosphere grid.
        val_logtemp: log of effective temperature
        val_logg: log of surface gravity
        val_mu: cos(angle) of angle of emission
        val_area: area of the surface element
        
        >>> self.Get_flux_nosum(val_logtemp, val_logg, val_mu, val_area)
        flux
        """
        grid = self.grid
        logtemp = self.logtemp
        logg = self.logg
        mu = self.mu
        w1temp, jtemp = self.Getaxispos(logtemp,val_logtemp)
        w1logg, jlogg = self.Getaxispos(logg,val_logg)
        w1mu, jmu = self.Getaxispos(mu,val_mu)
        flux = Utils.Grid.Interp_photometry_nosum(grid, w1temp, w1logg, w1mu, jtemp, jlogg, jmu, val_area, val_mu)
        return flux

    def Getaxispos(self, xx, x):
        """
        """
        if isinstance(x, (list, tuple, np.ndarray)):
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

    def Interp_orig(self, val_logtemp, val_logg, val_mu):
        """
        Obsolete!!!
        """
        grid = self.grid
        logtemp = self.logtemp
        logg = self.logg
        mu = self.mu
        w1temp, jtemp = self.Getaxispos(logtemp,val_logtemp)
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
        flux = np.exp(fl)*val_mu
        return flux

