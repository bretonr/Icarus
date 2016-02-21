# Licensed under a 3-clause BSD style license - see LICENSE

__all__ = ["Star"]

from ..Utils.import_modules import *
from .. import Utils
from .Star_base import Star_base

logger = logging.getLogger(__name__)


######################## class Star ########################
class Star(Star_base):
    """Star(Star_base)
    This class allows determine the flux of the companion star
    in a binary system using an atmosphere grid. It is derived
    from the Star_base class.
    
    The noticeable difference is that the surface is constructed
    from a geodesic tesselation of equilateral triangles derived
    from an isocahedron.

    Axis convention:
        x: From the primary center of mass towards the secondary.
        y: Along the orbital plane along the orbital motion.
        z: Along the orbital angular momentum.
    """
    def __init__(self, ndiv, atmo_grid=None, read=False, oldchi=False):
        Star_base.__init__(self, ndiv, atmo_grid=atmo_grid)
        if read:
            self._Read_geodesic()
        else:
            self._New_Initialization()
        self.oldchi = oldchi

    def _New_Initialization(self):
        """Initialization(self)
        Run important initialization steps important for the
        class to work.
        
        self.vertices contains x,y,z coordinates of vertices. shape = self.n_vertices,3
        self.faces contains indices of vertices forming faces. shape = self.n_faces,3
        self.assoc contains indices of faces associated to a vertice. shape = self.n_vertices,6
            Note: when only 5 faces associated, 6th value is equal to -99
        """
        print( "Generating the geodesic surface using PyGTS" )
        try:
            import gts
        except:
            print( "You likely don't have the PyGTS package installed on your computer." )
            print( "It is impossible to create the surface vertices from scratch." )
            print( "Will trying reading them from the restart file instead." )
            self._Read_geodesic()
            return
        # Generate the geodesic primitives
        s = gts.sphere(self.ndiv)
        x,y,z,t = gts.get_coords_and_face_indices(s,True)
        self.vertices = np.c_[x,y,z]
        self.faces = np.array(t)
        self.n_vertices = self.vertices.shape[0]
        self.n_faces = self.faces.shape[0]
        print( "Calculatating the associations" )
        self.assoc = Utils.Tessellation.Match_assoc(self.faces, self.n_vertices)
        
        # We will pre-calculate the surface areas. They will need to be multiplied by rc^2.
        # The calculation is simply the Pythagorean sum of the areas of the respective projections on the x,y,z planes.
        print( "meshing the surface" )
        mesh = self.vertices[self.faces]
        print( "calculating the area" )
        self.pre_area = 0.5 *np.sqrt( ((mesh[:,0,0]*mesh[:,1,1]+mesh[:,1,0]*mesh[:,2,1]+mesh[:,2,0]*mesh[:,0,1]) - (mesh[:,0,1]*mesh[:,1,0]+mesh[:,1,1]*mesh[:,2,0]+mesh[:,2,1]*mesh[:,0,0]))**2 + ((mesh[:,0,1]*mesh[:,1,2]+mesh[:,1,1]*mesh[:,2,2]+mesh[:,2,1]*mesh[:,0,2]) - (mesh[:,0,2]*mesh[:,1,1]+mesh[:,1,2]*mesh[:,2,1]+mesh[:,2,2]*mesh[:,0,1]))**2 + ((mesh[:,0,2]*mesh[:,1,0]+mesh[:,1,2]*mesh[:,2,0]+mesh[:,2,2]*mesh[:,0,0]) - (mesh[:,0,0]*mesh[:,1,2]+mesh[:,1,0]*mesh[:,2,2]+mesh[:,2,0]*mesh[:,0,2]))**2 )
        # The cosine of x,y,z for the center of the faces. shape = n_faces, 3
        print( "calculating the angles" )
        self.cosx, self.cosy, self.cosz = mesh.mean(axis=1).T
        return

    def _Initialization(self):
        """Initialization(self)
        Run important initialization steps important for the
        class to work.
        
        self.vertices contains x,y,z coordinates of vertices. shape = self.n_vertices,3
        self.faces contains indices of vertices forming faces. shape = self.n_faces,3
        self.assoc contains indices of faces associated to a vertice. shape = self.n_vertices,6
            Note: when only 5 faces associated, 6th value is equal to -99
        """
        print( "Generating the geodesic surface" )
        # Generate the geodesic primitives
        self.n_faces, self.n_vertices, self.faces, self.vertices, self.assoc = Utils.Tessellation.Make_geodesic(self.ndiv)
        # We will pre-calculate the surface areas. They will need to be multiplied by rc^2.
        # The calculation is simply the Pythagorean sum of the areas of the respective projections on the x,y,z planes.
        print( "meshing the surface" )
        mesh = self.vertices[self.faces]
        print( "calculating the area" )
        self.pre_area = 0.5 *np.sqrt( ((mesh[:,0,0]*mesh[:,1,1]+mesh[:,1,0]*mesh[:,2,1]+mesh[:,2,0]*mesh[:,0,1]) - (mesh[:,0,1]*mesh[:,1,0]+mesh[:,1,1]*mesh[:,2,0]+mesh[:,2,1]*mesh[:,0,0]))**2 + ((mesh[:,0,1]*mesh[:,1,2]+mesh[:,1,1]*mesh[:,2,2]+mesh[:,2,1]*mesh[:,0,2]) - (mesh[:,0,2]*mesh[:,1,1]+mesh[:,1,2]*mesh[:,2,1]+mesh[:,2,2]*mesh[:,0,1]))**2 + ((mesh[:,0,2]*mesh[:,1,0]+mesh[:,1,2]*mesh[:,2,0]+mesh[:,2,2]*mesh[:,0,0]) - (mesh[:,0,0]*mesh[:,1,2]+mesh[:,1,0]*mesh[:,2,2]+mesh[:,2,0]*mesh[:,0,2]))**2 )
        # The cosine of x,y,z for the center of the faces. shape = n_faces, 3
        print( "calculating the angles" )
        self.cosx, self.cosy, self.cosz = mesh.mean(axis=1).T
        return

    def Outline(self, ntheta=100, debug=False):
        """Outline(ntheta=100, debug=False)
        Calculates the radii of the outline of the star for a vector
        of theta=np.arange(ntheta)/ntheta*cts.TWOPI.
            theta is defined as np.arctan2(y_projected,z_projected).
            theta0 = 0
            dtheta = cts.TWOPI/ntheta
        
        ntheta (100): Number of points defining the outline.
        debug (False): Print debug information when True.
        
        >>> self._Outline()
        """
        if debug: print( 'Begin _Outline()' )
        
        theta = np.arange(ntheta, dtype=float)/ntheta * cts.TWOPI
        y = np.cos(theta)
        z = np.sin(theta)
        # radii of the outline of the star.
        radii = self._Radius(y*0., y, z, self.psi0, self.rc_eq)
        return radii

    def _Read_geodesic(self):
        """Read_geodesic()
        The information about the geodesic surface on the unit
        sphere has already been precalculated. We simply load the
        one have the desired precision.
        """
        #f = open('geodesic/geodesic_n%i.txt'%self.ndiv, 'r')
        f = open(Utils.__path__[0][:-5]+'geodesic/geodesic_n%i.txt'%self.ndiv, 'r')
        lines = f.readlines()
        
        # We store the number of vertices, faces and edges as class variables.
        tmp, self.n_vertices, self.n_faces, self.n_edges = lines[0].split()
        self.n_vertices = int(self.n_vertices)
        self.n_faces = int(self.n_faces)
        self.n_edges = int(self.n_edges)

        # Vertice information contains coordinate x,y,z of vertices. shape = n_vertices,3
        self.vertices = np.array([l.split() for l in lines[1:1+self.n_vertices]], dtype=float)

        # Face information contains indices of vertices forming faces. shape = n_faces,3
        self.faces = np.array([l.split() for l in lines[1+self.n_vertices:1+self.n_vertices+self.n_faces]], dtype=int)
        self.faces = self.faces[:,1:]
        
        # We calculate the associations
        self.assoc = Utils.Tessellation.Match_assoc(self.faces, self.n_vertices)
        
        # We will pre-calculate the surface areas. They will need to be multiplied by rc^2.
        # The calculation is simply the Pythagorean sum of the areas of the respective projections on the x,y,z planes.
        mesh = self.vertices[self.faces]
        self.pre_area = 0.5 *np.sqrt( ((mesh[:,0,0]*mesh[:,1,1]+mesh[:,1,0]*mesh[:,2,1]+mesh[:,2,0]*mesh[:,0,1]) - (mesh[:,0,1]*mesh[:,1,0]+mesh[:,1,1]*mesh[:,2,0]+mesh[:,2,1]*mesh[:,0,0]))**2 + ((mesh[:,0,1]*mesh[:,1,2]+mesh[:,1,1]*mesh[:,2,2]+mesh[:,2,1]*mesh[:,0,2]) - (mesh[:,0,2]*mesh[:,1,1]+mesh[:,1,2]*mesh[:,2,1]+mesh[:,2,2]*mesh[:,0,1]))**2 + ((mesh[:,0,2]*mesh[:,1,0]+mesh[:,1,2]*mesh[:,2,0]+mesh[:,2,2]*mesh[:,0,0]) - (mesh[:,0,0]*mesh[:,1,2]+mesh[:,1,0]*mesh[:,2,2]+mesh[:,2,0]*mesh[:,0,2]))**2 )
        # The cosine of x,y,z for the center of the faces. shape = n_faces, 3
        self.cosx, self.cosy, self.cosz = mesh.mean(axis=1).T
        return

    def Radius(self):
        """Radius()
        Returns the volume-averaged radius of the star, in
        units of orbital separation.
        
        >>> self.Radius()
        """
        return (self.rc**3).mean()**(1./3)

    def Roche(self):
        """Roche()
        Returns the volume-averaged Roche lobe radius
        of the star in units of orbital separation.

        For the geodesic tessellation, the volume-averaged
        Roche-lobe radius is easilly found since each surface
        element subtend the same solid angle. Therefore, the
        volume-averaged radius is the cubic root of the average
        values of the radii cubed <rc^3>^1/3.
        
        >>> self.Roche()
        """
        filling = self.filling
        self.Make_surface(filling=1.)
        radius = self.Radius()
        self.Make_surface(filling=filling)
        return radius

    def _Surface(self, debug=False):
        """_Surface(debug=False)
        Calculates the surface grid values of surface gravity
        and surface element area by solving the potential
        equation.
        
        debug (False): Print debug information when True.
        
        >>> self._Surface()
        """
        logger.log(9, "start")
        if debug: print( 'Begin _Surface()' )
        ## Calculate some quantities
        self._Calc_qp1by2om2()
        
        ## Saddle point, i.e. the Roche-lobe radius at L1 (on the near side)
        xl1 = self._Saddle(0.5)
        self.L1 = xl1
        if debug: print( 'Saddle %f' %xl1 )
        ## Potential at the saddle point, L1
        psil1 = self._Potential(xl1, 0., 0.)[-1]
        if debug: print( 'Potential psil1 %f' %psil1 )
        
        ## rc_l1 is the stellar radius on the near side, i.e. the nose of the star
        self.rc_l1 = self.filling*xl1
        if debug: print( 'rc_l1 %f' %self.rc_l1 )
        ## Potential at rc_l1, the nose of the star
        trc, trx, dpsi, dpsidx, dpsidy, dpsidz, psi0 = self._Potential(self.rc_l1, 0., 0.)
        self.psi0 = psi0
        if debug: print( 'Potential psi0\n trc: %f, trx %f, dpsi %f, dpsidx %f, dpsidy %f, dpsidz %f, psi0 %f' % (trc, trx, dpsi, dpsidx, dpsidy, dpsidz, self.psi0) )
        
        ## rc_pole is stellar radius at 90 degrees, i.e. at the pole, which is perpendicular to the line separating the two stars and the orbital plane
        if debug: print( 'psi0,r '+str(self.psi0)+' '+str(r) )
        self.rc_pole = self._Radius(0.,0.,1.,self.psi0,self.rc_l1)
        trc, trx, dpsi, dpsidx, dpsidy, dpsidz, psi = self._Potential(0.,0.,self.rc_pole)
        ## log surface gravity at the pole of the star
        self.logg_pole = np.log10(np.sqrt(dpsidx**2+dpsidy**2+dpsidz**2))
        
        ## rc_eq is stellar radius at 90 degrees in the orbital plane, i.e. at the equator, but not in the direction of the companion
        self.rc_eq = self._Radius(0.,1.,0.,self.psi0,self.rc_l1)
        trc, trx, dpsi, dpsidx, dpsidy, dpsidz, psi = self._Potential(0.,self.rc_eq,0.)
        ## log surface gravity at the pole of the star
        self.logg_eq = np.log10(np.sqrt(dpsidx**2+dpsidy**2+dpsidz**2))
        
        ## r_vertices are the radii of the vertices. shape = n_vertices
        self.r_vertices = self._Radius(self.vertices[:,0], self.vertices[:,1], self.vertices[:,2], self.psi0, self.rc_l1)
        
        ### Calculate useful quantities for all surface elements
        ## rc corresponds to r1 from Tjemkes et al., the distance from the center of mass of the pulsar companion. shape = n_faces
        self.rc = self._Radius(self.cosx, self.cosy, self.cosz, self.psi0, self.rc_l1)
        ## rx corresponds to r2 from Tjemkes et al., the distance from the center of mass of the pulsar. shape = n_faces
        trc, self.rx, dpsi, dpsidx, dpsidy, dpsidz, psi = self._Potential(self.rc*self.cosx,self.rc*self.cosy,self.rc*self.cosz)
        ## log surface gravity. shape = n_faces
        geff = self._Geff(dpsidx, dpsidy, dpsidz)
        self.logg = np.log10(geff)
        ## gradient of the gravitational potential in x,y,z. shape = n_faces
        self.gradx = -dpsidx/geff
        self.grady = -dpsidy/geff
        self.gradz = -dpsidz/geff
        ## coschi is the cosine angle between the rx and the surface element. shape = n_faces
        ## A value of 1 means that the companion's surface element is directly facing the pulsar, 0 is at the limb and -1 on the back.
        ## The following is the old way, which is derived from the spherical approximation, i.e. that the normal to the surface is approximately the same as the radial position 
        if self.oldchi:
            self.coschi = -(self.rc-self.cosx)/self.rx
        ## The better calculation should use the gradient as the normal vector, and the direction to the pulsar as positive x.
        ## This implies that the angle coschi is simply the x component of the gradient.
        else:
            self.coschi = self.gradx.copy()
    
        ## surface area. shape = n_faces
        self.area = self.rc**2 * self.pre_area
        logger.log(9, "end")
        return

######################## class Star ########################

