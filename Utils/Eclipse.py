# Licensed under a 3-clause BSD style license - see LICENSE

import os

import scipy.weave

from .import_modules import *

# Try to import the Shapely package
try:
    import shapely.geometry
    import shapely.speedups
    import shapely.prepared
    shapely.speedups.enable()
    _HAS_SHAPELY = True
except:
    print( "The Shapely package cannot be imported. This will run normally but not eclipse optimization can be used." )
    _HAS_SHAPELY = False


##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##
## Contain functions to perform eclipse calculations
##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##


def Hsr(y1, z1, y2, z2, faces):
    """Hsr(y1, z1, y2, z2, faces)
    Hidden surface removal algorithm.
    Returns the weight of each face/surface element with
    0, 1/3, 2/3, 1, going from not covered to fully covered.
    
    y1,2, z1,2: Projected coordinates of the stars in the sky
        plane (y is along the orbital plane, z is along the orbital
        angular momentum).
    faces: List of faces of the primary (containing the
        vertice indices).
    
    >>> weights = Hsr(y1, z1, y2, z2, faces)
    """
    # In the following, we loop through the vertices of the occulted primary
    # and determine whether it lies within a surface element of the secondary.
    # The index of the occulted vertices are stored in an array.
    inds = []
    for i in np.arange(y1.size):
        dy = (y1[i]-y2)
        dz = (z1[i]-z2)
        dr2 = (dy**2+dz**2)
        k = dr2.argsort()[:3]
        if inside_triangle([y1[i],z1[i]], [y2[k[0]],z2[k[0]]], [y2[k[1]],z2[k[1]]], [y2[k[2]],z2[k[2]]]):
            inds.append(i)
    # Calculate the weights of the different surface elements.
    # 0, 1/3, 2/3 or 1, with 1 being all three vertices occulted.
    weights = 0.
    for i in inds:
        weights += (faces == i).mean(1)
    return weights

def Hsr_c(faces_b, vertices_b, r_vertices_b, assoc_b, faces_f, vertices_f, r_vertices_f, assoc_f, incl, orbph, q, rmax_f, rmin_f):
    """Hsr_c(faces_b, vertices_b, r_vertices_b, assoc_b, faces_f, vertices_f, r_vertices_f, assoc_f, incl, orbph, q, rmax_f, rmin_f)
    Hidden surface removal algorithm (implemented in C).
    Returns the weight of each face/surface element with
    0, 1/3, 2/3, 1, going from not covered to fully covered.
    
    >>> weights = Hsr_c(y1, z1, y2, z2, faces)
    """
    support_code = """
    #include <iostream>
    // method to convert 3d coordinate to sky plane projection
    void to_skyplane( double x, double y, double z, double incl, double orbph, double offsety, double offsetz, double *ynew, double *znew ) {
        double cos_incl, sin_incl, cos_phs, sin_phs, xnew;
        
        cos_incl = cos(incl);
        sin_incl = sin(incl);
        cos_phs = cos(orbph);
        sin_phs = sin(orbph);
        xnew = x*cos_phs + y*sin_phs;
        *ynew = -x*sin_phs + y*cos_phs + offsety;
        *znew = z*sin_incl + xnew*cos_incl + offsetz;
        //*ynew = 5.;
        //*znew = 10.;
        //std::cout << "Test" << std::endl;
        //std::cout << *ynew << std::endl;
        //std::cout << *znew << std::endl;

    }
    
    // function that returns true if the point py,pz lies inside
    // the triangle described by y1,2,3 and z1,2,3
    bool inside_triangle( double py, double pz, double y1, double z1, double y2, double z2, double y3, double z3 ) {
        double detT,lambda1,lambda2,lambda3;
        
        detT = (y1-y3)*(z2-z3) - (z1-z3)*(y2-y3);
        lambda1 = ((z2-z3)*(py-y3) - (y2-y3)*(pz-z3)) / detT;
        lambda2 = (-(z1-z3)*(py-y3) + (y1-y3)*(pz-z3)) / detT;
        lambda3 = 1 - lambda1 - lambda2;
        return (0. <= lambda1) && (lambda1 <= 1.) && (0. <= lambda2) && (lambda2 <= 1.) && (0. <= lambda3) and (lambda3 <= 1.);
    }
    """
    
    code = """
    
    //FILE * pfile;
    //pfile = fopen("debug.txt","w");
    
    double PI = 4 * atan(1);
    
    double phs_b = orbph*2*PI;
    double phs_f = (orbph+0.5)*2*PI;
    
    // calculate the offset of the eclipsing star
    double offsety_f, offsetz_f;
    to_skyplane( 1/(1+q), 0., 0., incl, phs_f, 0., 0., &offsety_f, &offsetz_f );
    
    //fprintf(pfile, "Offset front: %f %f\\n",offsety_f,offsetz_f);
    //printf("Offset front: %f %f\\n",offsety_f,offsetz_f);
    
    // Vy,Vz: sky plane coordinates of eclipsing star vertices
    double Vy[n_vertices_f], Vz[n_vertices_f];

    // loop through the vertices of the eclipsing star to convert to sky plane
    for (int i=0; i<n_vertices_f; i++) {
        to_skyplane( vertices_f(i,0)*r_vertices_f(i),vertices_f(i,1)*r_vertices_f(i),vertices_f(i,2)*r_vertices_f(i),incl,phs_f,offsety_f,offsetz_f,&Vy[i],&Vz[i] );
    }
    // loop through the vertices of the eclipsing star to convert to sky plane
    
    // calculate the offset of the eclipsed star
    double offsety_b, offsetz_b;
    to_skyplane( q/(1+q), 0., 0., incl, phs_b, 0., 0., &offsety_b, &offsetz_b );
    
    //fprintf(pfile, "Offset back: %f %f\\n",offsety_b,offsetz_b);
    //printf("Offset back: %f %f\\n",offsety_b,offsetz_b);
    
    double vx, vy, vz;
    double y, z;
    double dist2, dr2;
    double y1, z1, y2, z2, y3, z3;
    int id_vertice_f, id_surface_b, id_surface_f, id;
    // loop through the vertices of the eclipsed star
    for (int i=0; i<n_vertices_b; i++) {
        // first, retrieve the vertice coordinates
        vx = vertices_b(i,0);
        vy = vertices_b(i,1);
        vz = vertices_b(i,2);
        // transform the vertice coordinates to the sky plane
        to_skyplane( vx*r_vertices_b(i),vy*r_vertices_b(i),vz*r_vertices_b(i),incl,phs_b,offsety_b,offsetz_b,&y,&z );
        
        //fprintf(pfile, "  Vertice %i: (x,y,z) %f %f %f; (sky y, sky z) [%f], [%f]\\n",i,vx,vy,vz,y,z);
        //printf("  Vertice %i: (x,y,z) %f %f %f; (sky y, sky z) [%f], [%f]\\n",i,vx,vy,vz,y,z);
        
        // optimize the occlusion algorithm
        dr2 = pow(y-offsety_f,2) + pow(z-offsetz_f,2);
        
        // if the point is further than the maximum extent of the
        // eclipsing star, we just step to the next surface element.
        if (dr2 > pow(rmax_f,2)) {
            //fprintf(pfile, "    Continue (clearly outside!)");
            //printf("    Continue (clearly outside!)");
            continue;
        
        // if the next point is within the minimum extent of the
        // eclipsing star, it is necessarily hidden so we add the
        // weights right away.
        } else if (dr2 < pow(rmin_f,2)) {
            // for each face that the vertice belongs to
            for (int m=0; m<6; m++) {
                id_surface_b = assoc_b(i,m);
                //fprintf(pfile, "    Tagging id_surface_b %i\\n",id_surface_b);
                //printf("    Tagging id_surface_b %i\\n",id_surface_b);
                if (id_surface_b >= 0) {
                    weight(id_surface_b) += 1.;
                }
            }
            // for each face that the vertice belongs to
        
        // if none of the above conditions is met, we have to go
        // through the lengthy calculation.
        } else {
        
            // identify the nearest vertice of the eclipsing star
            // for each vertice of the eclipsing star
            id = 0;
            dist2 = 10.;
            for (int l=0; l<n_vertices_f; l++) {
                dr2 = pow(y-Vy[l],2) + pow(z-Vz[l],2);
                //fprintf(pfile, "  l %i; Vy: %f, Vz: %f, dr2: %f\\n",l,Vy[l],Vz[l],dr2);
                if (dr2 < dist2) {
                    id = l;
                    dist2 = dr2;
                }
            }
            // for each vertice of the eclipsing star
            
            //fprintf(pfile, "    Nearest front vertice %i at %f\\n",id,dist2);
            //printf("    Nearest front vertice %i at %f\\n",id,dist2);
            
            // once we know the nearest vertice of the eclipsing star
            // to the vertice of the eclipsed star, we check if the latter
            // lies inside one of its associated surfaces. If it does, we
            // add one unit to the weight of that face.
            // for each associated face
            for (int k=0; k<6; k++) {
                id_surface_f = assoc_f(id,k);
                // if the id < 0 (i.e. -99), the vertice has no 6th
                // associated surface so we continue the loop with the
                // next surface id
                if (id_surface_f < 0) continue;
                id_vertice_f = faces_f(id_surface_f,0);
                y1 = Vy[id_vertice_f];
                z1 = Vz[id_vertice_f];
                id_vertice_f = faces_f(id_surface_f,1);
                y2 = Vy[id_vertice_f];
                z2 = Vz[id_vertice_f];
                id_vertice_f = faces_f(id_surface_f,2);
                y3 = Vy[id_vertice_f];
                z3 = Vz[id_vertice_f];
                //fprintf(pfile, "    Nearest vertice %i: id_surface_f %i (%i); [%f, %f, %f], [%f, %f, %f]\\n",id,id_surface_f,k,y1,y2,y3,z1,z2,z3);
                //printf("    Nearest vertice %i: id_surface_f %i (%i); [%f, %f, %f], [%f, %f, %f]\\n",id,id_surface_f,k,y1,y2,y3,z1,z2,z3);
                // if vertice is hidden
                if (inside_triangle( y,z,y1,z1,y2,z2,y3,z3 )) {
                    // if the vertice is hidden, add 1 to the weight
                    // of each face that it belongs to
                    // for each face that the vertice belongs to
                    for (int m=0; m<6; m++) {
                        id_surface_b = assoc_b(i,m);
                        if (id_surface_b >= 0) {
                            weight(id_surface_b) += 1.;
                            //fprintf(pfile, "      Hiding %i (%i), weight %f\\n",id_surface_b,m,weight(id_surface_b));
                            //printf("      Hiding %i (%i), weight %f\\n",id_surface_b,m,weight(id_surface_b));
                        }
                    }
                    // for each face that the vertice belongs to
                    // unnecessary to loop further
                    k = 6;
                }
                // if vertice is hidden
            }
            // for each associated face
        }
        // optimize the occlusion algorithm
    }
    // loop through the vertices of the eclipsed star

    // loop through the faces of the eclipsed star to normalize the weights
    for (int i=0; i<n_faces_b; i++) {
        // we normalize and subtract so that a fully covered face
        // has a weight of 0 and a non-covered one 1.
        //printf("%i %f",i,weight(i));
        weight(i) = 1-weight(i)/3.;
        //printf("  %f\\n",weight(i));
    }
    // loop through the faces of the eclipsed star to normalize the weights
    
    
    //fclose(pfile);
    
    """
    n_vertices_b = vertices_b.shape[0]
    n_vertices_f = vertices_f.shape[0]
    n_faces_b = faces_b.shape[0]
    weight = np.zeros(n_faces_b, dtype=np.float)
    #extra_compile_args = extra_link_args = ['-O3 -fopenmp']
    extra_compile_args = extra_link_args = ['']
    tmp = scipy.weave.inline(code, ['vertices_b', 'r_vertices_b', 'assoc_b', 'faces_f', 'vertices_f', 'r_vertices_f', 'assoc_f', 'n_vertices_b', 'n_vertices_f', 'n_faces_b', 'incl', 'orbph', 'q', 'rmax_f', 'rmin_f', 'weight'], type_converters=scipy.weave.converters.blitz, compiler='gcc', support_code=support_code, extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=['<cstdio>', '<cmath>', '<omp.h>'], verbose=1)
    return weight

def Inside_triangle(p, a, b, c):
    """ inside_triangle(p, a, b, c)
    p: point (x,y)
    a, b, c: vertices of the triangle (x,y)
    
    >>> inside_triangle(p, a, b, c)
    """
    detT = (a[0]-c[0])*(b[1]-c[1]) - (a[1]-c[1])*(b[0]-c[0])
    lambda1 = ((b[1]-c[1])*(p[0]-c[0]) - (b[0]-c[0])*(p[1]-c[1])) / detT
    lambda2 = (-(a[1]-c[1])*(p[0]-c[0]) + (a[0]-c[0])*(p[1]-c[1])) / detT
    lambda3 = 1 - lambda1 - lambda2
    return (0 < lambda1 < 1) and (0 < lambda2 < 1) and (0 < lambda3 < 1)

def Occultation_approx(vertices, r_vertices, assoc, n_faces, incl, orbph, q, ntheta, radii):
    """Occultation_approx(vertices, r_vertices, assoc, n_faces, incl, orbph, q, ntheta, radii)

    Hidden surface removal algorithm.
    Returns the weight of each face/surface element with
    0, 1, 2, 3; going from not covered to fully covered.
    """
    support_code = """
    #include <iostream>
    // method to convert 3d coordinate to sky plane projection
    void to_skyplane( double x, double y, double z, double incl, double orbph, double offsety, double offsetz, double *ynew, double *znew ) {
        double cos_incl, sin_incl, cos_phs, sin_phs, xnew;
        
        cos_incl = cos(incl);
        sin_incl = sin(incl);
        cos_phs = cos(orbph);
        sin_phs = sin(orbph);
        xnew = x*cos_phs + y*sin_phs;
        *ynew = -x*sin_phs + y*cos_phs + offsety;
        *znew = z*sin_incl + xnew*cos_incl + offsetz;
        //*ynew = 5.;
        //*znew = 10.;
        //std::cout << "Test" << std::endl;
        //std::cout << *ynew << std::endl;
        //std::cout << *znew << std::endl;
    }
    """
    
    code = """
    double tmp_y, tmp_z;
    double offsety, offsetz;
    
    to_skyplane( -1./(1.+q), 0., 0., incl, orbph, 0., 0., &offsety, &offsetz );
    //std::cout << offsety << " " << offsetz << std::endl;
    tmp_y = offsety;
    tmp_z = offsetz;
    to_skyplane( q/(1.+q), 0., 0., incl, orbph, 0., 0., &offsety, &offsetz );
    //std::cout << offsety << " " << offsetz << std::endl;
    offsety -= tmp_y;
    offsetz -= tmp_z;
    
    #pragma omp parallel shared(n_vertices,vertices,assoc,r_vertices,incl,orbph,weight,ntheta,radii,offsety,offsetz) default(none)
    {
    int ind;
    double vx, vy, vz;
    double y, z;
    double theta, w, r;
    int pos;
    #pragma omp for
    for (int i=0; i<n_vertices; i++) {
        // first, retrieve the vertice coordinates
        vx = vertices(i,0);
        vy = vertices(i,1);
        vz = vertices(i,2);
        // transform the vertice coordinates to the sky plane
        to_skyplane( vx*r_vertices(i),vy*r_vertices(i),vz*r_vertices(i),incl,orbph,offsety,offsetz,&y,&z );
        
        theta = atan2(z,y);
        pos = int( theta/ntheta );
        w = theta/ntheta - pos;
        //std::cout << "Test" << std::endl;
        //std::cout << theta << " " << pos << " " << w << std::endl;
        r = radii(pos)*(1-w) + radii(pos+1)*w;
        
        if ((pow(y,2)+pow(z,2)) < pow(r,2)) {
            for (int j=0; j<5; j++) {
                ind = assoc(i, j);
                weight(ind) += 1.;
            }
            if (assoc(i, 5) != -99) {
                ind = assoc(i, 5);
                weight(ind) += 1.;
            }
        }
    }
    }
    """
    q = np.float(q)
    n_vertices = vertices.shape[0]
    weight = np.zeros(n_faces, dtype=np.float)
    try:
        if os.uname()[0] == 'Darwin':
            extra_compile_args = extra_link_args = ['-O3']
        else:
            extra_compile_args = extra_link_args = ['-O3 -fopenmp']
        tmp = scipy.weave.inline(code, ['n_vertices', 'vertices', 'assoc', 'r_vertices', 'incl', 'orbph', 'q', 'weight', 'ntheta', 'radii'], type_converters=scipy.weave.converters.blitz, compiler='gcc', support_code=support_code, extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=['<cstdio>', '<cmath>', '<omp.h>'], verbose=2, force=0)
    except:
        tmp = scipy.weave.inline(code, ['n_vertices', 'vertices', 'assoc', 'r_vertices', 'incl', 'orbph', 'q', 'weight', 'ntheta', 'radii'], type_converters=scipy.weave.converters.blitz, compiler='gcc', support_code=support_code, extra_compile_args=['-O3'], extra_link_args=['-O3'], headers=['<cstdio>', '<cmath>'], verbose=2, force=0)
    
    return weight

def Occultation_shapely(vertices, faces_ind, incl, orbph, q, ntheta, radii):
    """Occultation_shapely(vertices, faces_ind, incl, orbph, q, ntheta, radii)

    Hidden surface removal algorithm.
    Returns the weight of each face/surface element (i.e.
    fractional area uncovered).
    
    vertices (array (3,n_vertices)): Array of vertices making the faces of the
        star located in front.
    faces_ind (array (n_faces,3)): Array providing the vertice indices of the
        faces of the star located in front.
    """
    # Making sure that shapely is installed
    if not _HAS_SHAPELY:
        print( "You must install the Shapely package to run this function." )
        return
    
    #print( "orbph: {}".format(orbph) )
    import time
    T = []
    T.append(time.time())
    
    # Defining the front star polygon
    theta = np.arange(ntheta, dtype=float)/ntheta * cts.TWOPI
    xoff, yoff = Observer_2Dprojection(1./(1+q), 0., 0., incl, orbph+0.5)
    x_front = radii * np.cos(theta) + xoff
    y_front = radii * np.sin(theta) + yoff
    star_front = shapely.geometry.Polygon(np.c_[x_front, y_front].copy())
    prepared_star_front = shapely.prepared.prep(star_front)
    T.append(time.time())
    #print( "T{}: {} ({})".format(len(T), T[-1]-T[0], T[-1]-T[-2]) )
    
    # Defining the faces of the back star
    x_back, y_back = Observer_2Dprojection(vertices[0], vertices[1], vertices[2], incl, orbph, xoffset=q/(1.+q))
    x_back = x_back[faces_ind]
    y_back = y_back[faces_ind]
    faces = np.array([shapely.geometry.Polygon(zip(*xy)) for xy in zip(x_back,y_back)])
    T.append(time.time())
    #print( "T{}: {} ({})".format(len(T), T[-1]-T[0], T[-1]-T[-2]) )

    # Calculating the indices of overlapping, partially hidden and fully hidden faces
    overlap = np.array([prepared_star_front.intersects(f) for f in faces])
    partial = overlap.copy()
    if overlap.any():
        hidden = np.array([prepared_star_front.contains(f) for f in faces[overlap]])
        if hidden.any():
            partial[overlap] = ~hidden
            hidden = overlap - partial
        else:
            hidden = np.zeros_like(overlap)
    T.append(time.time())
    #print( "T{}: {} ({})".format(len(T), T[-1]-T[0], T[-1]-T[-2]) )

    # Calculating the weights (fractional hidden area
    weights = np.ones_like(overlap, dtype=float)
    weights[overlap] = 0.
    if partial.any():
        partial_weight = 1 - np.array( [ star_front.intersection(face).area/face.area for face in faces[partial] ] )
        weights[partial] = partial_weight
    T.append(time.time())
    #print( "T{}: {} ({})".format(len(T), T[-1]-T[0], T[-1]-T[-2]) )

    # Calculating the total area in two different ways
    area_geo = np.array([face.area for face in faces])
    #area_ica = np.abs(star2.area * star2.cosx)

    # Printing useful information
    #print( "area_geo.sum() {}".format(area_geo.sum()) )
    #print( "area_ica.sum() {}".format(area_ica.sum()) )
    #print( "fraction eclipse {}".format((weights*area_geo).sum()/area_geo.sum()) )
    #print( "predicted fraction eclipse {}".format( 1 - (star1.Radius()/star2.Radius())**2 ) )
    T.append(time.time())
    #print( "T{}: {} ({})".format(len(T), T[-1]-T[0], T[-1]-T[-2]) )

    # Plotting
    #from Pgplot import *
    #nextplotpage()
    #plotxy(y_back.flat, x_back.flat, line=None, symbol=1, aspect=1, rangey=[-1.1,1.1], rangex=[-1.1,1.1])
    #x, y = star_front.exterior.xy
    #plotxy(y, x, color=2)
    #plotxy([-2.,2.],[0.,0.])
    #plotxy([0.,0.],[-2.,2.])

    return weights

def Observer_2Dprojection(x, y, z, incl, orbph, xoffset=None):
    """ Observer_2Dprojection(x, y, z, incl, orbph, xoffset=None)
    x, y, z: cartesian coordinates
    incl: orbital inclination (radians)
    orbph: orbital phase (0-1)
    xoffset (None): x offset, due to star not located at the
        origin of the coordinate system
    
    >>> new_y,new_z = Observer_2Dprojection(x, y, z, incl, orbph, xoffset=None)
    """
    orbph = orbph%1
    cos_incl = np.cos(incl)
    sin_incl = np.sin(incl)
    cos_phs = np.cos(orbph*cts.TWOPI)
    sin_phs = np.sin(orbph*cts.TWOPI)
    xnew = x*cos_phs + y*sin_phs
    ynew = -x*sin_phs + y*cos_phs
    znew = z*sin_incl + xnew*cos_incl
    # We want to allow for a shift so that we translate the coordinate system from
    # the star center to the barycenter.
    if xoffset is not None:
        yoff, zoff = Observer_2Dprojection(xoffset, 0., 0., incl, orbph)
        ynew += yoff
        znew += zoff
    return ynew, znew

def Observer_3Dprojection(x, y, z, incl, orbph, xoffset=None):
    """ Observer_3Dprojection(x, y, z, incl, orbph, xoffset=None)
    x, y, z: cartesian coordinates
    incl: orbital inclination (radians)
    orbph: orbital phase (0-1)
    xoffset (None): x offset, due to star not located at the
        origin of the coordinate system
    
    >>> new_x,new_y,new_z = Observer_3Dprojection(x, y, z, incl, orbph, xoffset=None)
    """
    orbph = orbph%1
    cos_incl = np.cos(incl)
    sin_incl = np.sin(incl)
    cos_phs = np.cos(orbph*cts.TWOPI)
    sin_phs = np.sin(orbph*cts.TWOPI)
    xnew = x*cos_phs + y*sin_phs
    ynew = -x*sin_phs + y*cos_phs
    znew = z
    z = znew*sin_incl + xnew*cos_incl
    x = -znew*cos_incl + xnew*sin_incl
    y = ynew
    # We want to allow for a shift so that we translate the coordinate system from
    # the star center to the barycenter.
    if xoffset is not None:
        xoff, yoff, zoff = Observer_3Dprojection(xoffset, 0., 0., incl, orbph)
        x += xoff
        y += yoff
        z += zoff
    return x, y, z

def Overlap(y1, z1, y2, z2):
    """ overlap(y1, z1, y2, z2)
    """
    y1min = y1.min()
    y1max = y1.max()
    y2min = y2.min()
    y2max = y2.max()
    r1 = (y1max-y1min)
    r2 = (y2max-y2min)
    ycenter1 = (y1min+y1max)*0.5
    ycenter2 = (y2min+y2max)*0.5
    z1min = z1.min()
    z1max = z1.max()
    z2min = z2.min()
    z2max = z2.max()
    zcenter1 = (z1min+z1max)*0.5
    zcenter2 = (z2min+z2max)*0.5
    # Determining which points of star 1 and 2 are potentially overlapping
    # We approximate that the stars are confined within circles
    inds1 = np.sqrt((y1 - ycenter2)**2 + (z1 - zcenter2)**2) < r2
    inds2 = np.sqrt((y2 - ycenter1)**2 + (z2 - zcenter1)**2) < r1
    return inds1, inds2

def System_2Dprojection(x1, y1, z1, x2, y2, z2, incl, orbph, q):
    """ system_2Dprojection(x1, y1, z1, x2, y2, z2, incl, orbph, q)
    x1,2, y1,2, z1,2: cartesian coordinates
    incl: orbital inclination (radians)
    orbph: orbital phase of primary (0-1)
    mass ratio: M2/M1, used to calculate the x offset
    
    >>> new_y1,new_z1,new_y2,new_z2 = system_2Dprojection(x1, y1, z1, x2, y2, z3, incl, orbph, q)
    """
    y1, z1 = Observer_2Dprojection(x1, y1, z1, incl, orbph, -q/(1+q))
    y2, z2 = Observer_2Dprojection(x2, y2, z2, incl, orbph+0.5, 1/(1+q))
    return y1, z1, y2, z2

def Weights_transit(inds_highres, weight_highres, n_lowres):
    """Weights_transit(inds_highres, weight_highres, n_lowres)
    
    """
    code = """
    #pragma omp parallel shared(n_highres,weight_lowres,weight_highres,inds_highres) default(none)
    {
    int ind;
    #pragma omp for
    for (int i=0; i<n_highres; i++) {
        ind = inds_highres(i);
        weight_lowres(ind) += weight_highres(i);
    }
    }
    """
    
    n_highres = inds_highres.shape[0]
    weight_lowres = np.zeros(n_lowres, dtype='float')
    try:
        if os.uname()[0] == 'Darwin':
            extra_compile_args = extra_link_args = ['-O3']
        else:
            extra_compile_args = extra_link_args = ['-O3 -fopenmp']
        get_assoc = scipy.weave.inline(code, ['n_highres', 'weight_lowres', 'weight_highres', 'inds_highres'], type_converters=scipy.weave.converters.blitz, compiler='gcc', extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, headers=['<omp.h>'], libraries=['m'], verbose=2, force=0)
    except:
        get_assoc = scipy.weave.inline(code, ['n_highres', 'weight_lowres', 'weight_highres', 'inds_highres'], type_converters=scipy.weave.converters.blitz, compiler='gcc', extra_compile_args=['-O3'], extra_link_args=['-O3'], libraries=['m'], verbose=2, force=0)
    return weight_lowres




