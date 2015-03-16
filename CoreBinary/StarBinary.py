# Licensed under a 3-clause BSD style license - see LICENSE

__all__ = ["StarBinary"]

from ..Utils.import_modules import *
from .. import Utils
from .. import Core
from ..Utils import Eclipse


######################## class StarBinary ########################
class StarBinary(object):
    """StarBinary
    This class allows determine the flux of the two stars in a
    binary system using an atmosphere grid.
    
    The idea is the following:
        Some basic parameters of the binary system are provided (mass ratio, 
        corotation factor, filling factor, companion temperature, irradiated
        energy (spin down) from the primary). From these, the potential equation
        is solved and a grid of [temperature - surface gravity] is calculated.
        Then, provided an orbital inclination and an orbital phase, the cos(emission
        angle to observer) is calculated. For each (T, g, mu) triplet, we can infer
        the radiance from an atmosphere grid, which, after multiplicating by the
        surface element of that it covers, provides the contribution to the total
        luminosity of the star. It is taken into account that some energy received
        from the primary heat up the exposed side of the companion.
        
        The flux for each star is calculated. The primary first, and then the
        secondary, by simply inverting the mass ratio and calculating the
        appropriate distance.
    
    The phases are such the star 1 is at superior conjunction at phase 0.5
    (hence possibly eclipsed by star 2) and at inferior conjunction at phase 0
    (hence possibly eclipsing star 2).
    """
    def __init__(self, ndiv1, ndiv2, atmo_grid=None, read=False):
        """__init__
        Initialize the class instance.
        
        ndiv: The number of surface element subdivisions. Defines how
            coarse/fine the surface grid is.
            Can be a two-element array (e.g. [5,8]), which would define the
            regular and the high-resolution level. If defined this way, the
            high-resolution level will be used in the partial eclipse case.
        atmo_grid (None): An atmosphere model grid from which the
            radiance is interpolated.
        read (False): If true, will read the geodesic primitives instead of
            generating them from scratch.
        
        It is optional to provide an atmosphere grid. If none is provided, it will
        have to be passed as a parameter to the routine calculating the flux.
        
        >>> lightcurve = StarBinary(nafl1, ndiv2)
        """
        # We define some useful quantities.
        # We set the class attributes
        if atmo_grid is not None:
           self.atmo_grid = atmo_grid
        
        print( "Instantiating the primary star" )
        # Single resolution for the primary
        if isinstance(ndiv1, int):
            print( " ...working at a single resolution" )
            self.ndiv1 = ndiv1
            self.primary = Core.Star(self.ndiv1, atmo_grid=atmo_grid, read=read)
            self.primary_hd = None
        # Dual resolution for the primary
        elif len(ndiv1) == 2:
            print( " ...working at a dual resolution" )
            self.ndiv1, self.ndiv1_hd = ndiv1
            self.primary = Core.Star(self.ndiv1, atmo_grid=atmo_grid, read=read)
            # If the resolution of the second argument is smaller or equal to the first argument, there will be no high resolution
            if self.ndiv1_hd <= self.ndiv1:
                print( " ...not quite, working at a single resolution" )
                self.primary_hd = None
            # If high resolution is required, we will keep track of the association between the high resolution triangle and the low resolution ones
            else:
                print( " ...calculating the surface subsampling" )
                if read:
                    self.primary_hd = Core.Star(self.ndiv1_hd, atmo_grid=atmo_grid, read=read)
                    #self.ind_subsampling1 = np.loadtxt('geodesic/ind_subsampling_n%i_n%i.txt'%(self.ndiv1, self.ndiv1_hd), dtype='int')
                    self.ind_subsampling1 = np.loadtxt(Utils.__path__[0][:-5]+'geodesic/ind_subsampling_n%i_n%i.txt'%(self.ndiv1, self.ndiv1_hd), dtype='int')
                else:
                    triangle_assoc = []
                    x, y, z = self.primary.cosx, self.primary.cosy, self.primary.cosz
                    # For each extra subdivision, we calculate the surface and determine the association between the triangle of this refinement level and the one just before
                    for i in np.arange(self.ndiv1,self.ndiv1_hd)+1:
                        print( " ...subsampling from %s to %s" %(i-1,i) )
                        self.primary_hd = Core.Star(i, atmo_grid=atmo_grid, read=read)
                        x_high, y_high, z_high = self.primary_hd.cosx, self.primary_hd.cosy, self.primary_hd.cosz
                        triangle_assoc.append( Utils.Tessellation.Match_triangles(x_high, y_high, z_high, x, y, z) )
                        x, y, z = x_high, y_high, z_high
                    # Now we go backward and associate the triangles from the highest resolution down to the lowest resolution. Results are stored in self.ind_subsampling.
                    print( " ...almost done" )
                    self.ind_subsampling1 = triangle_assoc.pop()
                    while triangle_assoc:
                        print( " ...associations..." )
                        self.ind_subsampling1 = Utils.Tessellation.Match_subtriangles(self.ind_subsampling1, triangle_assoc.pop())
                # We also store the total weight, which is 3 vertices * 4**(ndiv_hd-ndiv) for the normalization
                self.total_weight1 = 3 * 4**(self.ndiv1_hd-self.ndiv1)
        # In case of problem for the primary's resolution
        else:
            print( "Problem with ndiv1. Has to be a float or two-element array" )
        
        print( "Instantiating the secondary star" )
        # Single resolution for the secondary
        if isinstance(ndiv2, int):
            print( " ...working at a single resolution" )
            self.ndiv2 = ndiv2
            self.secondary = Core.Star(self.ndiv2, atmo_grid=atmo_grid, read=read)
            self.secondary_hd = None
        # Dual resolution for the secondary
        elif len(ndiv2) == 2:
            print( " ...working at a dual resolution" )
            self.ndiv2, self.ndiv2_hd = ndiv2
            self.secondary = Core.Star(self.ndiv2, atmo_grid=atmo_grid, read=read)
            # If the resolution of the second argument is smaller or equal to the first argument, there will be no high resolution
            if self.ndiv2_hd <= self.ndiv2:
                print( " ...not quite, working at a single resolution" )
                self.secondary_hd = None
            # If high resolution is required, we will keep track of the association between the high resolution triangle and the low resolution ones
            else:
                print( " ...calculating the surface subsampling" )
                if read:
                    self.secondary_hd = Core.Star(self.ndiv2_hd, atmo_grid=atmo_grid, read=read)
                    #self.ind_subsampling2 = np.loadtxt('geodesic/ind_subsampling_n%i_n%i.txt'%(self.ndiv2, self.ndiv2_hd), dtype='int')
                    self.ind_subsampling2 = np.loadtxt(Utils.__path__[0][:-5]+'geodesic/ind_subsampling_n%i_n%i.txt'%(self.ndiv2, self.ndiv2_hd), dtype='int')
                else:
                    triangle_assoc = []
                    x, y, z = self.secondary.cosx, self.secondary.cosy, self.secondary.cosz
                    # For each extra subdivision, we calculate the surface and determine the association between the triangle of this refinement level and the one just before
                    for i in np.arange(self.ndiv2,self.ndiv2_hd)+1:
                        print( " ...subsampling from %s to %s" %(i-1,i) )
                        self.secondary_hd = Core.Star(i, atmo_grid=atmo_grid, read=read)
                        x_high, y_high, z_high = self.secondary_hd.cosx, self.secondary_hd.cosy, self.secondary_hd.cosz
                        triangle_assoc.append( Utils.Tessellation.Match_triangles(x_high, y_high, z_high, x, y, z) )
                        x, y, z = x_high, y_high, z_high
                    # Now we go backward and associate the triangles from the highest resolution down to the lowest resolution. Results are stored in self.ind_subsampling.
                    print( " ...almost done" )
                    self.ind_subsampling2 = triangle_assoc.pop()
                    while triangle_assoc:
                        print( " ...associations..." )
                        self.ind_subsampling2 = Utils.Tessellation.Match_subtriangles(self.ind_subsampling2, triangle_assoc.pop())
                # We also store the total weight, which is 3 vertices * 4**(ndiv_hd-ndiv) for the normalization
                self.total_weight2 = 3 * 4**(self.ndiv2_hd-self.ndiv2)
        # In case of problem for the secondary's resolution
        else:
            print( "Problem with ndiv2. Has to be a float or two-element array" )
        # This is the end of the initialization function
        return

    def Flux(self, phase, atmo_grid=None, nosum=False):
        """Flux(phase, atmo_grid=None, nosum=False)
        Return the flux interpolated from the atmosphere grid.
        
        phase: orbital phase (in orbital fraction; 0: companion 
            in front, 0.5: companion behind).
        atmo_grid (optional): atmosphere grid instance used to
            calculate the flux.
        nosum (False): if true, will no sum across the surface
            and returns (fsum1, fsum2).
        
        >>> self.Flux(phase)
        flux
        """
        if atmo_grid is None:
            atmo_grid = self.atmo_grid
        fsum1 = self.primary.Flux(phase, atmo_grid=atmo_grid, nosum=nosum)
        fsum2 = self.secondary.Flux(phase+0.5, atmo_grid=atmo_grid, nosum=nosum)
        if nosum:
            return fsum1, fsum2
        return fsum1+fsum2

    def Flux_eclipse_old(self, phase, atmo_grid=None, ntheta=100, doppler1=0., doppler2=0.):
        """Flux_eclipse(phase, atmo_grid=None, ntheta=100, doppler1=0., doppler2=0.)
        Return the flux interpolated from the atmosphere grid.
        
        phase: orbital phase (in orbital fraction; 0: companion 
            in front, 0.5: companion behind).
        atmo_grid (optional): atmosphere grid instance used to
            calculate the flux.
        ntheta (100): number of points defining the outline of the
            eclipsing star.
        doppler (0.): coefficient for the Doppler boosting. If 0., no
            Doppler boosting is performed. If None, will use the value
            returned by self.primary.Doppler_boosting() or
            self.secondary.Doppler_boosting().
        
        >>> self.Flux_eclipse(phase)
        flux
        """
        phase = phase%1
        
        if atmo_grid is None:
            atmo_grid = self.atmo_grid
        
        type1, type2 = self.Occultation(phase)
        
        if type1 == "full":
            fsum1 = 0.
        elif type1 == "partial":
            radii = self.secondary.Outline(ntheta)
            weights1 = Eclipse.Occultation_approx(self.primary.vertices, self.primary.r_vertices, self.primary.assoc, self.primary.n_faces, self.primary.incl, phase*cts.TWOPI, self.primary.q, ntheta, radii)
            mu = self.primary._Mu(phase)
            inds = (mu>0)*(weights1<3)
            fsum1 = self.primary.Flux(phase, atmo_grid=atmo_grid, nosum=True, mu=mu, inds=inds, doppler=doppler1)
            fsum1 *= 1 - weights1[inds]/3
            fsum1 = fsum1.sum() * self.normalize1
        elif type1 == "partial_hd":
            radii = self.secondary.Outline(ntheta)
            weights1 = Eclipse.Occultation_approx(self.primary_hd.vertices, self.primary_hd.r_vertices, self.primary_hd.assoc, self.primary_hd.n_faces, self.primary_hd.incl, phase*cts.TWOPI, self.primary_hd.q, ntheta, radii)
            mu = self.primary_hd._Mu(phase)
            inds = (mu>0)*(weights1<3)
            fsum1 = self.primary_hd.Flux(phase, atmo_grid=atmo_grid, nosum=True, mu=mu, inds=inds, doppler=doppler1)
            fsum1 *= 1 - weights1[inds]/3
            fsum1 = fsum1.sum()
        else:
            fsum1 = self.primary.Flux(phase, atmo_grid=atmo_grid, nosum=False, doppler=doppler1) * self.normalize1
        
        if type2 == "full":
            fsum2 = 0.
        elif type2 == "partial":
            radii = self.primary.Outline(ntheta)
            weights2 = Eclipse.Occultation_approx(self.secondary.vertices, self.secondary.r_vertices, self.secondary.assoc, self.secondary.n_faces, self.secondary.incl, ((phase+0.5)%1)*cts.TWOPI, self.secondary.q, ntheta, radii)
            mu = self.secondary._Mu((phase+0.5)%1)
            inds = (mu>0)*(weights2<3)
            fsum2 = self.secondary.Flux((phase+0.5)%1, atmo_grid=atmo_grid, nosum=True, mu=mu, inds=inds, doppler=doppler2)
            fsum2 *= 1 - weights2[inds]/3
            fsum2 = fsum2.sum() * self.normalize2
        elif type2 == "partial_hd":
            radii = self.primary.Outline(ntheta)
            weights2 = Eclipse.Occultation_approx(self.secondary_hd.vertices, self.secondary_hd.r_vertices, self.secondary_hd.assoc, self.secondary_hd.n_faces, self.secondary_hd.incl, ((phase+0.5)%1)*cts.TWOPI, self.secondary_hd.q, ntheta, radii)
            mu = self.secondary_hd._Mu((phase+0.5)%1)
            inds = (mu>0)*(weights2<3)
            fsum2 = self.secondary_hd.Flux((phase+0.5)%1, atmo_grid=atmo_grid, nosum=True, mu=mu, inds=inds, doppler=doppler2)
            fsum2 *= 1 - weights2[inds]/3
            fsum2 = fsum2.sum()
        else:
            fsum2 = self.secondary.Flux((phase+0.5)%1, atmo_grid=atmo_grid, nosum=False, doppler=doppler2) * self.normalize2
        
        return fsum1+fsum2#, fsum1, fsum2

    def Flux_eclipse_shapely(self, phase, atmo_grid=None, ntheta=100, doppler1=0., doppler2=0., nosum=False, invert=True):
        """Flux_eclipse_shapely(phase, atmo_grid=None, ntheta=100, doppler1=0., doppler2=0., nosum=False, invert=True)
        Return the flux interpolated from the atmosphere grid.
        
        Uses the outline of the eclipsing star in order to speed up computation.
        
        phase: orbital phase (in orbital fraction; 0: companion 
            in front, 0.5: companion behind).
        atmo_grid (optional): atmosphere grid instance used to
            calculate the flux.
        ntheta (100): number of points defining the outline of the
            eclipsing star.
        doppler (0.): coefficient for the Doppler boosting. If 0., no
            Doppler boosting is performed. If None, will use the value
            returned by self.primary.Doppler_boosting() or
            self.secondary.Doppler_boosting().
        nosum (False): if true, will no sum across the surface
            and returns (fsum1, fsum2).
        invert (True): If true, will use the high resolution computation
            for the out-of-eclipse and the low resolution for the eclipse.
        
        >>> self.Flux_eclipse(phase)
        flux
        """
        phase = phase%1
        
        if atmo_grid is None:
            atmo_grid = self.atmo_grid
        
        #import time
        #t0 = time.time()
        
        type1, type2 = self.Occultation(phase)
        
        #t1 = time.time()
        #print( "phase: {}".format(phase) )
        if type1 == "full":
            fsum1 = 0.
            #print( "    full phase: {}".format(phase) )
        elif type1.find("partial") != -1:
            if type1 == "partial" or invert:
                #print( "partial1" )
                radii = self.secondary.Outline(ntheta)
                vertices = self.primary.vertices.T * self.primary.r_vertices
                mu = self.primary._Mu(phase)
                inds =  (mu>0).nonzero()[0]
                weights1 = Eclipse.Occultation_shapely(vertices, self.primary.faces[inds], self.primary.incl, phase, self.primary.q, ntheta, radii)
                inds1 = weights1>0
                inds = inds[inds1]
                weights1 = weights1[inds1]
                fsum1 = self.primary.Flux(phase, atmo_grid=atmo_grid, nosum=True, mu=mu, inds=inds, doppler=doppler1)
                fsum1 *= weights1
                fsum1 = fsum1.sum() * self.normalize1
                #print( "    partial phase: {}".format(phase) )
            else:
                #print( "partial_hd1" )
                radii = self.secondary.Outline(ntheta)
                vertices = self.primary_hd.vertices.T * self.primary_hd.r_vertices
                mu = self.primary_hd._Mu(phase)
                inds = (mu>0).nonzero()[0]
                weights_highres = Eclipse.Occultation_shapely(vertices, self.primary_hd.faces[inds], self.primary_hd.incl, phase, self.primary_hd.q, ntheta, radii)
                weights1 = Eclipse.Weights_transit(self.ind_subsampling1[inds], weights_highres, self.primary.n_faces) / (self.total_weight1/3.)
                mu = self.primary._Mu(phase)
                inds = (mu>0)*(weights1>0)
                fsum1 = self.primary.Flux(phase, atmo_grid=atmo_grid, nosum=True, mu=mu, inds=inds, doppler=doppler1)
                fsum1 *= weights1[inds]
                fsum1 = fsum1.sum() * self.normalize1
                #print( "    partial HD phase: {}".format(phase) )
        else:
            if self.primary_hd is None or not invert:
                #print( "out1" )
                fsum1 = self.primary.Flux(phase, atmo_grid=atmo_grid, nosum=False, doppler=doppler1) * self.normalize1
            else:
                #print( "out hd1" )
                fsum1 = self.primary_hd.Flux(phase, atmo_grid=atmo_grid, nosum=False, doppler=doppler1) #* self.normalize1 ## not needed here
            #print( "    regular phase: {}".format(phase) )
        
        if type2 == "full":
            fsum2 = 0.
            #print( "    full phase: {}".format(phase) )
        elif type2.find("partial") != -1:
            if type2 == "partial" or invert:
                #print( "partial2" )
                radii = self.primary.Outline(ntheta)
                vertices = self.secondary.vertices.T * self.secondary.r_vertices
                mu = self.secondary._Mu((phase+0.5)%1)
                inds = (mu>0).nonzero()[0]
                weights2 = Eclipse.Occultation_shapely(vertices, self.secondary.faces[inds], self.secondary.incl, ((phase+0.5)%1), self.secondary.q, ntheta, radii)
                inds2 = weights2>0
                inds = inds[inds2]
                weights2 = weights2[inds2]
                fsum2 = self.secondary.Flux((phase+0.5)%1, atmo_grid=atmo_grid, nosum=True, mu=mu, inds=inds, doppler=doppler2)
                fsum2 *= weights2
                fsum2 = fsum2.sum() * self.normalize2
                #print( "    partial phase: {}".format(phase) )
            else:
                #print( "partial_hd2" )
                radii = self.primary.Outline(ntheta)
                vertices = self.secondary_hd.vertices.T * self.secondary_hd.r_vertices
                mu = self.secondary_hd._Mu((phase+0.5)%1)
                inds = (mu>0).nonzero()[0]
                weights_highres = Eclipse.Occultation_shapely(vertices, self.secondary_hd.faces[inds], self.secondary_hd.incl, ((phase+0.5)%1), self.secondary_hd.q, ntheta, radii)
                weights2 = Eclipse.Weights_transit(self.ind_subsampling2[inds], weights_highres, self.secondary.n_faces) / (self.total_weight2/3.)
                mu = self.secondary._Mu((phase+0.5)%1)
                inds = (mu>0)*(weights2>0)
                fsum2 = self.secondary.Flux((phase+0.5)%1, atmo_grid=atmo_grid, nosum=True, mu=mu, inds=inds, doppler=doppler2)
                fsum2 *= weights2[inds]
                fsum2 = fsum2.sum() * self.normalize2
                #print( "    partial HD phase: {}".format(phase) )
        else:
            if self.secondary_hd is None or not invert:
                #print( "out2" )
                fsum2 = self.secondary.Flux((phase+0.5)%1, atmo_grid=atmo_grid, nosum=False, doppler=doppler2) * self.normalize2
            else:
                #print( "out hd2" )
                fsum2 = self.secondary_hd.Flux((phase+0.5)%1, atmo_grid=atmo_grid, nosum=False, doppler=doppler2) #* self.normalize2 ## not needed here
            #print( "    regular phase: {}".format(phase) )
        
        #t2 = time.time()
        #print( "        time: {} {}".format(t1-t0, t2-t1) )
        
        if nosum:
            return fsum1, fsum2
        return fsum1+fsum2

    def Flux_eclipse(self, phase, atmo_grid=None, ntheta=100, doppler1=0., doppler2=0., nosum=False):
        """Flux_eclipse(phase, atmo_grid=None, ntheta=100, doppler1=0., doppler2=0., nosum=False)
        Return the flux interpolated from the atmosphere grid.
        
        Uses the outline of the eclipsing star in order to speed up computation.
        
        phase: orbital phase (in orbital fraction; 0: companion 
            in front, 0.5: companion behind).
        atmo_grid (optional): atmosphere grid instance used to
            calculate the flux.
        ntheta (100): number of points defining the outline of the
            eclipsing star.
        doppler (0.): coefficient for the Doppler boosting. If 0., no
            Doppler boosting is performed. If None, will use the value
            returned by self.primary.Doppler_boosting() or
            self.secondary.Doppler_boosting().
        nosum (False): if true, will no sum across the surface
            and returns (fsum1, fsum2).
        
        >>> self.Flux_eclipse(phase)
        flux
        """
        phase = phase%1
        
        if atmo_grid is None:
            atmo_grid = self.atmo_grid
        
        type1, type2 = self.Occultation(phase)
        
        if type1 == "full":
            fsum1 = 0.
        elif type1 == "partial":
            radii = self.secondary.Outline(ntheta)
            weights1 = Eclipse.Occultation_approx(self.primary.vertices, self.primary.r_vertices, self.primary.assoc, self.primary.n_faces, self.primary.incl, phase*cts.TWOPI, self.primary.q, ntheta, radii)
            mu = self.primary._Mu(phase)
            inds = (mu>0)*(weights1<3)
            fsum1 = self.primary.Flux(phase, atmo_grid=atmo_grid, nosum=True, mu=mu, inds=inds, doppler=doppler1)
            fsum1 *= 1 - weights1[inds]/3
            fsum1 = fsum1.sum() * self.normalize1
        elif type1 == "partial_hd":
            radii = self.secondary.Outline(ntheta)
            weights_highres = Eclipse.Occultation_approx(self.primary_hd.vertices, self.primary_hd.r_vertices, self.primary_hd.assoc, self.primary_hd.n_faces, self.primary_hd.incl, phase*cts.TWOPI, self.primary_hd.q, ntheta, radii)
            weights1 = Eclipse.Weights_transit(self.ind_subsampling1, weights_highres, self.primary.n_faces)
            mu = self.primary._Mu(phase)
            inds = (mu>0)*(weights1<self.total_weight1)
            fsum1 = self.primary.Flux(phase, atmo_grid=atmo_grid, nosum=True, mu=mu, inds=inds, doppler=doppler1)
            fsum1 *= 1 - weights1[inds]/self.total_weight1
            fsum1 = fsum1.sum() * self.normalize1
        else:
            fsum1 = self.primary.Flux(phase, atmo_grid=atmo_grid, nosum=False, doppler=doppler1) * self.normalize1
        
        if type2 == "full":
            fsum2 = 0.
        elif type2 == "partial":
            radii = self.primary.Outline(ntheta)
            weights2 = Eclipse.Occultation_approx(self.secondary.vertices, self.secondary.r_vertices, self.secondary.assoc, self.secondary.n_faces, self.secondary.incl, ((phase+0.5)%1)*cts.TWOPI, self.secondary.q, ntheta, radii)
            mu = self.secondary._Mu((phase+0.5)%1)
            inds = (mu>0)*(weights2<3)
            fsum2 = self.secondary.Flux((phase+0.5)%1, atmo_grid=atmo_grid, nosum=True, mu=mu, inds=inds, doppler=doppler2)
            fsum2 *= 1 - weights2[inds]/3
            fsum2 = fsum2.sum() * self.normalize2
        elif type2 == "partial_hd":
            radii = self.primary.Outline(ntheta)
            weights_highres = Eclipse.Occultation_approx(self.secondary_hd.vertices, self.secondary_hd.r_vertices, self.secondary_hd.assoc, self.secondary_hd.n_faces, self.secondary_hd.incl, ((phase+0.5)%1)*cts.TWOPI, self.secondary_hd.q, ntheta, radii)
            weights2 = Eclipse.Weights_transit(self.ind_subsampling2, weights_highres, self.secondary.n_faces)
            mu = self.secondary._Mu((phase+0.5)%1)
            inds = (mu>0)*(weights2<self.total_weight2)
            fsum2 = self.secondary.Flux((phase+0.5)%1, atmo_grid=atmo_grid, nosum=True, mu=mu, inds=inds, doppler=doppler2)
            fsum2 *= 1 - weights2[inds]/self.total_weight2
            fsum2 = fsum2.sum() * self.normalize2
        else:
            fsum2 = self.secondary.Flux((phase+0.5)%1, atmo_grid=atmo_grid, nosum=False, doppler=doppler2) * self.normalize2
        
        if nosum:
            return fsum1, fsum2
        return fsum1+fsum2

    def Flux_doppler(self, phase, atmo_grid=None, velocity1=0., velocity2=0.):
        """Flux_doppler(phase, atmo_grid=None, velocity1=0., velocity2=0.)
        Return the flux interpolated from the atmosphere grid.
        Takes into account the Doppler shift of the different surface
        elements due to the orbital velocity.
        
        phase: orbital phase (in orbital fraction; 0: companion 
            in front, 0.5: companion behind).
        atmo_grid (optional): atmosphere grid instance used to
            calculate the flux.
        velocity1,2 (optional): extra velocity in m/s to be added.
        
        >>> self.Flux_doppler(phase)
        flux
        """
        if atmo_grid is None:
            atmo_grid = self.atmo_grid
        fsum1 = self.primary.Flux_doppler(phase, atmo_grid=atmo_grid, nosum=nosum, velocity=velocity1)
        fsum2 = self.secondary.Flux_doppler(phase+0.5, atmo_grid=atmo_grid, nosum=nosum, velocity=velocity2)
        return fsum1+fsum2

    def Make_surface(self, q=None, omega1=None, omega2=None, filling1=None, filling2=None, temp1=None, temp2=None, tempgrav1=None, tempgrav2=None, tirr1=None, tirr2=None, porb=None, k1=None, incl=None, normalize=True):
        """Make_surface(q=None, omega=None, filling=None, temp=None, tempgrav=None, tirr=None, porb=None, k1=None, incl=None)
        Provided some basic parameters about the binary system,
        calculates the surface grid values for surface gravity,
        temperature, surface area, etc.
        
        This function is just a conveniance function that calls
        internal functions. If q, omega and filling are not changed,
        only the temperature is recalculated.
        
        q (None): mass ratio (M2/M1)
        omega1,2 (None): co-rotation factor
        filling1,2 (None): filling factor
        temp1,2 (None): surface temperature of the star.
        tempgrav1,2 (None): gravity darkening coefficient.
        tirr1,2 (None): irradiation temperature.
            (lirr = eff * edot / (4*PI * a**2 * sigma))
            (tirr = (eff * edot / (4*PI * a**2 * sigma))**0.25)
        porb (None): orbital period, in seconds.
        k1 (None): velocity semi-amplitude, in m/s.
            v2sini is automatically calculated as q*k1
        incl (None): orbital inclination, in radians.
        normalize (True): If true, will calculate the normalization factor
            so that the two resolutions (normal and high resolution) match
            in terms of flux.
        
        >>> self.Make_surface(q, omega1, omega2, filling1, filling2, temp1, temp2, tempgrav1, tempgrav2, tirr1, tirr2, porb, k1, incl)
        """
        # Making the surface of the primary
        self.primary.Make_surface(q=q, omega=omega1, filling=filling1, temp=temp1, tempgrav=tempgrav1, tirr=tirr1, porb=porb, k1=k1, incl=incl)
        self.normalize1 = 1.
        if self.primary_hd:
            self.primary_hd.Make_surface(q=q, omega=omega1, filling=filling1, temp=temp1, tempgrav=tempgrav1, tirr=tirr1, porb=porb, k1=k1, incl=incl)
            if normalize:
                self.normalize1 = self.primary_hd.Flux(0.5) / self.primary.Flux(0.5)
        
        # Making the surface of the secondary
        self.secondary.Make_surface(q=1/q, omega=omega2, filling=filling2, temp=temp2, tempgrav=tempgrav2, tirr=tirr2, porb=porb, k1=k1/q, incl=incl)
        self.normalize2 = 1.
        if self.secondary_hd:
            self.secondary_hd.Make_surface(q=1/q, omega=omega2, filling=filling2, temp=temp2, tempgrav=tempgrav2, tirr=tirr2, porb=porb, k1=k1/q, incl=incl)
            if normalize:
                self.normalize2 = self.secondary_hd.Flux(0.5) / self.secondary.Flux(0.5)
        
        self.r1max = self.primary.rc.max()
        self.r2max = self.secondary.rc.max()
        self.r1min = self.primary.rc.min()
        self.r2min = self.secondary.rc.min()
        # Determining if the two stars will ever overlap in the sky plane
        self.overlap = (abs(np.cos(incl)) - self.r1max - self.r2max) < 0
        # If an overlap is possible, we calculate the orbital phase around which this should happened
        if self.overlap:
            #self.overlap_phs = np.arcsin(self.r1max + self.r2max)/cts.TWOPI
            cosi2 = np.cos(incl)**2
            sini2 = np.sin(incl)**2
            ## From Phoebe Science guide
            #self.overlap_phs = scipy.optimize.newton(lambda phs: np.sqrt(cosi2*np.cos(phs)**2+np.sin(phs)**2) - self.r1max - self.r2max, 0.05*cts.TWOPI) / cts.TWOPI
            ## From Kallrath and Milone
            #print( "incl: {}, r1max: {}, r2max: {}, r1min: {}, r2min: {}".format(incl, self.r1max, self.r2max, self.r1min, self.r2min) )
            self.overlap_phs = scipy.optimize.newton(lambda phs: np.sqrt(cosi2+sini2*np.sin(phs)**2) - self.r1max - self.r2max, 0.05*cts.TWOPI) / cts.TWOPI
            # Determining if a total eclipse will ever happen
            if self.r1min < self.r2min:
                self.full_eclipse1 = (np.cos(incl) + self.r1max - self.r2min) < 0
                self.full_eclipse2 = False
                if self.full_eclipse1:
                    ## From Phoebe Science guide
                    #self.full_eclipse_phs1 = scipy.optimize.newton(lambda phs: np.sqrt(cosi2*np.cos(phs)**2+np.sin(phs)**2) + self.r1max - self.r2max, self.overlap_phs) / cts.TWOPI
                    ## From Kallrath and Milone
                    self.full_eclipse_phs1 = scipy.optimize.newton(lambda phs: np.sqrt(cosi2+sini2*np.sin(phs)**2) + self.r1max - self.r2min, self.overlap_phs) / cts.TWOPI
                self.full_eclipse_phs2 = None
            else:
                self.full_eclipse2 = (np.cos(incl) + self.r2max - self.r1min) < 0
                self.full_eclipse1 = False
                if self.full_eclipse2:
                    ## From Phoebe Science guide
                    #self.full_eclipse_phs2 = scipy.optimize.newton(lambda phs: np.sqrt(cosi2*np.cos(phs)**2+np.sin(phs)**2) + self.r2max - self.r1max, self.overlap_phs) / cts.TWOPI
                    ## From Kallrath and Milone
                    self.full_eclipse_phs2 = scipy.optimize.newton(lambda phs: np.sqrt(cosi2+sini2*np.sin(phs)**2) + self.r2max - self.r1min, self.overlap_phs) / cts.TWOPI
                self.full_eclipse_phs1 = None
        else:
            self.overlap_phs = None
            self.full_eclipse1 = False
            self.full_eclipse2 = False
            self.full_eclipse_phs1 = None
            self.full_eclipse_phs2 = None
        #print( self.overlap_phs, self.full_eclipse1, self.full_eclipse2, self.full_eclipse_phs1, self.full_eclipse_phs2 )
        return

    def Occultation(self, phase, debug=False):
        """Occultation(self, phase)
        Given an orbital phase, calculates the type of occultation
        for each star.
        
        phase: the orbital phase (in the range [0,1]).
        
        Returns for each star:
            "none": Fully visible
            "full": Fully eclipsed
            "partial": Partially eclipsed
            "partial_hd": Partially eclipsed (high definition surface available)
        """
        # if no overlap is possible
        if self.overlap is None or self.overlap == False:
            type1 = "none"
            type2 = "none"
            if debug: print( "No overlap possible in this system." )
        
        # (phase <= 0.25 and phase >= 0.75) the secondary star is in the back
        elif ((phase <= self.overlap_phs) or (phase >= 1-self.overlap_phs)):
            type1 = "none"
            # Full eclipse of the secondary
            if self.full_eclipse2 and ((phase <= self.full_eclipse_phs2) or (phase >= 1-self.full_eclipse_phs2)):
                type2 = "full"
                if debug: print( "Full eclipse of secondary." )
            # Partial eclipse of the secondary
            else:
                # if a high resolution secondary exists
                if self.secondary_hd:
                    type2 = "partial_hd"
                else:
                    type2 = "partial"
                if debug: print( "Partial eclipse of secondary" )
        
        # (0.25 <= phase <= 0.75) the primary star is in the back
        elif ((0.5-self.overlap_phs) <= phase <= (0.5+self.overlap_phs)):
            type2 = "none"
            # Full eclipse of the primary
            if self.full_eclipse1 and ((0.5-self.full_eclipse_phs1) <= phase <= (0.5+self.full_eclipse_phs1)):
                type1 = "full"
                if debug: print( "Full eclipse of primary." )
            # Partial eclipse of the primary
            else:
                # if a high resolution primary exists
                if self.primary_hd:
                    type1 = "partial_hd"
                else:
                    type1 = "partial"
                if debug: print( "Partial eclipse of primary." )
        
        # if overlap is possible but not in the range for it
        else:
            type1 = "none"
            type2 = "none"
            if debug: print( "No overlap at this phase." )
        
        return type1, type2

######################## class StarBinary ########################

