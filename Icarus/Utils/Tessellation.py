# Licensed under a 3-clause BSD style license - see LICENSE

import scipy.weave

from .import_modules import *


##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##
## Tessellation utilities
## Contain functions that pertain to "tessellation-related"
## purposes such as calculating triangle associations,
## generating vertice primitives, etc.
##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##


def Make_geodesic(n):
    """ Make_geodesic(n)
    Makes the primitives of a geodesic surface based on an
    isocahedron which is subdivided n times in smaller triangles.
    Return the number of vertices, surfaces, associations and
    their related vectors.
    
    n: integer number of subdivisions (can be zero)
    
    >>> n_faces, n_vertices, myfaces, myvertices, myassoc = Make_geodesic(n)
    """
    support_code = """
    static int n_vertices;
    static int n_faces;
    static int n_edges;
    static float *vertices = NULL;
    static int *faces = NULL;
    static int *assoc = NULL;
    
    static int edge_walk; 
    static int *start = NULL; 
    static int *end = NULL; 
    static int *midpoint = NULL; 
    
    static void 
    init_icosahedron (void) 
    { 
        float t = (1+sqrt(5))/2;
        float tau = t/sqrt(1+t*t);
        float one = 1/sqrt(1+t*t);
        
        float icosahedron_vertices[] = 
        {tau, one, 0.0,
        -tau, one, 0.0,
        -tau, -one, 0.0,
        tau, -one, 0.0,
        one, 0.0 ,  tau,
        one, 0.0 , -tau,
        -one, 0.0 , -tau,
        -one, 0.0 , tau,
        0.0 , tau, one,
        0.0 , -tau, one,
        0.0 , -tau, -one,
        0.0 , tau, -one};
        
        int icosahedron_faces[] = 
        {4, 8, 7,
        4, 7, 9,
        5, 6, 11,
        5, 10, 6,
        0, 4, 3,
        0, 3, 5,
        2, 7, 1,
        2, 1, 6,
        8, 0, 11,
        8, 11, 1,
        9, 10, 3,
        9, 2, 10,
        8, 4, 0,
        11, 0, 5,
        4, 9, 3,
        5, 3, 10,
        7, 8, 1,
        6, 1, 11,
        7, 2, 9,
        6, 10, 2};
        
        n_vertices = 12;
        n_faces = 20;
        n_edges = 30;
        
        vertices = (float*)malloc(3*n_vertices*sizeof(float));
        faces = (int*)malloc(3*n_faces*sizeof(int));
        memcpy ((void*)vertices, (void*)icosahedron_vertices, 3*n_vertices*sizeof(float));
        memcpy ((void*)faces, (void*)icosahedron_faces, 3*n_faces*sizeof(int));
    }
    
    static int 
    search_midpoint (int index_start, int index_end) 
    { 
        int i;
        for (i=0; i<edge_walk; i++) 
            if ((start[i] == index_start && end[i] == index_end) || 
	        (start[i] == index_end && end[i] == index_start)) 
                {
	            int res = midpoint[i];
                
	            /* update the arrays */
	            start[i]    = start[edge_walk-1];
	            end[i]      = end[edge_walk-1];
	            midpoint[i] = midpoint[edge_walk-1];
	            edge_walk--;
	            
	            return res; 
                }
            
            /* vertex not in the list, so we add it */
            start[edge_walk] = index_start;
            end[edge_walk] = index_end; 
            midpoint[edge_walk] = n_vertices; 
            
            /* create new vertex */ 
            vertices[3*n_vertices]   = (vertices[3*index_start] + vertices[3*index_end]) / 2.0;
            vertices[3*n_vertices+1] = (vertices[3*index_start+1] + vertices[3*index_end+1]) / 2.0;
            vertices[3*n_vertices+2] = (vertices[3*index_start+2] + vertices[3*index_end+2]) / 2.0;
            
            /* normalize the new vertex */ 
            float length = sqrt (vertices[3*n_vertices] * vertices[3*n_vertices] +
        		       vertices[3*n_vertices+1] * vertices[3*n_vertices+1] +
        		       vertices[3*n_vertices+2] * vertices[3*n_vertices+2]);
            length = 1/length;
            vertices[3*n_vertices] *= length;
            vertices[3*n_vertices+1] *= length;
            vertices[3*n_vertices+2] *= length;
            
            n_vertices++;
            edge_walk++;
            return midpoint[edge_walk-1];
    }
    
    static void 
    subdivide (void) 
    { 
        int n_vertices_new = n_vertices+2*n_edges; 
        int n_faces_new = 4*n_faces; 
        int i; 
        
        edge_walk = 0;
        n_edges = 2*n_vertices + 3*n_faces; 
        start = (int*)malloc(n_edges*sizeof (int)); 
        end = (int*)malloc(n_edges*sizeof (int)); 
        midpoint = (int*)malloc(n_edges*sizeof (int)); 
        
        int *faces_old = (int*)malloc (3*n_faces*sizeof(int)); 
        faces_old = (int*)memcpy((void*)faces_old, (void*)faces, 3*n_faces*sizeof(int)); 
        vertices = (float*)realloc ((void*)vertices, 3*n_vertices_new*sizeof(float)); 
        faces = (int*)realloc ((void*)faces, 3*n_faces_new*sizeof(int)); 
        n_faces_new = 0; 
        
        for (i=0; i<n_faces; i++) 
        { 
            int a = faces_old[3*i]; 
            int b = faces_old[3*i+1]; 
            int c = faces_old[3*i+2]; 
            
            int ab_midpoint = search_midpoint (b, a);
            int bc_midpoint = search_midpoint (c, b);
            int ca_midpoint = search_midpoint (a, c);
            
            faces[3*n_faces_new] = a; 
            faces[3*n_faces_new+1] = ab_midpoint; 
            faces[3*n_faces_new+2] = ca_midpoint; 
            n_faces_new++; 
            faces[3*n_faces_new] = ca_midpoint; 
            faces[3*n_faces_new+1] = ab_midpoint; 
            faces[3*n_faces_new+2] = bc_midpoint; 
            n_faces_new++; 
            faces[3*n_faces_new] = ca_midpoint; 
            faces[3*n_faces_new+1] = bc_midpoint; 
            faces[3*n_faces_new+2] = c; 
            n_faces_new++; 
            faces[3*n_faces_new] = ab_midpoint; 
            faces[3*n_faces_new+1] = b; 
            faces[3*n_faces_new+2] = bc_midpoint; 
            n_faces_new++; 
        } 
        n_faces = n_faces_new; 
        free (start); 
        free (end); 
        free (midpoint); 
        free (faces_old); 
    } 
    
    static void 
    associativity (void) 
    { 
        //printf ("associativity 2\\n");
        int i;
        
        assoc = (int*)malloc(6*n_vertices*sizeof(int)); 
        
        for (int v=0; v<n_vertices; v++)
        {
            i = 0;
            for (int f=0; f<n_faces; f++)
            {
                if ((faces[3*f] == v) || (faces[3*f+1] == v) || (faces[3*f+2] == v)) {
                    assoc[6*v+i] = f;
                    i += 1;
                }
            }
            if (i==5) {
                assoc[6*v+i] = -99;
                i = 6;
            }
        }
    }
    
    static void 
    isocahedron (int n_subdivisions)
    {
        int i;
        
        init_icosahedron ();
        
        for (i=0; i<n_subdivisions; i++)
            subdivide ();
        
        associativity ();
    }
    
    static void
    free_memory ()
    {
        if (vertices) free (vertices);
        if (faces) free (faces);
        if (assoc) free (assoc);
    }
    """
    
    code = """
    isocahedron(n);
    
    //printf ( "\\nisocahedron 1 \\n" );
    //printf ( "long %zu\\n", sizeof(long) );
    //printf ( "myfaces %zu\\n", sizeof(myfaces(0,0)) );
    //printf ( "faces %zu\\n", sizeof(faces[0]) );
    //printf ( "myvertices %zu\\n", sizeof(myvertices(0,0)) );
    //printf ( "vertices %zu\\n", sizeof(vertices[0]) );
    
    for (int i=0; i<n_faces; i++) {
        myfaces(i,0) = faces[3*i];
        myfaces(i,1) = faces[3*i+1];
        myfaces(i,2) = faces[3*i+2];
    }
    
    for (int i=0; i<n_vertices; i++) {
        myvertices(i,0) = vertices[3*i];
        myvertices(i,1) = vertices[3*i+1];
        myvertices(i,2) = vertices[3*i+2];
        
        myassoc(i,0) = assoc[6*i];
        myassoc(i,1) = assoc[6*i+1];
        myassoc(i,2) = assoc[6*i+2];
        myassoc(i,3) = assoc[6*i+3];
        myassoc(i,4) = assoc[6*i+4];
        myassoc(i,5) = assoc[6*i+5];
    }
    
    free_memory();
    """
    n_faces = 20 * 4**n
    myfaces = np.empty((n_faces,3), dtype=np.int)
    n_vertices = 2 + 10 * 4**n
    myvertices = np.empty((n_vertices,3), dtype=np.float)
    myassoc = np.empty((n_vertices,6), dtype=np.int)
    get_axispos = scipy.weave.inline(code, ['n','myfaces','myvertices','myassoc'], type_converters=scipy.weave.converters.blitz, compiler='gcc', libraries=['m'], verbose=2, support_code=support_code, force=0)
    return n_faces, n_vertices, myfaces, myvertices, myassoc

def Match_assoc(faces, n_vertices):
    """
    Match_assoc(faces, n_vertices)
    
    Returns the list of faces associated with each vertice.
    There are 5 or 6 faces per vertice, if 5, the 6th is -99.
    
    >>> assoc = Match_assoc(faces, n_vertices)
    """
    code = """
    int ind = 0;
    for (int i=0; i<n_faces; i++) {
        for (int j=0; j<3; j++) {
            ind = faces(i, j);
            for (int k=0; k<6; k++) {
                if (assoc(ind, k)  == -99) {
                    assoc(ind, k) = i;
                    break;
                }
            }
        }
    }
    """
    n_faces = faces.shape[0]
    assoc = -99 * np.ones((n_vertices,6), dtype=np.int)
    get_assoc = scipy.weave.inline(code, ['n_faces','faces','assoc'], type_converters=scipy.weave.converters.blitz, compiler='gcc', libraries=['m'], verbose=2, force=0)
    return assoc

def Match_triangles(high_x, high_y, high_z, low_x, low_y, low_z):
    """Match_triangles(high_x, high_y, high_z, low_x, low_y, low_z)
    
    The idea is to identify the triangles of the high resolution tessellation
    that belong to the low resolution version. Because we use a subdivision
    algorithm, which splits each triangle into 4 smaller triangles, there 
    should be 4**(n_highres - n_lowres) triangles associated with each low
    resolution one.
    
    Returns the list of low resolution face indices associated with each
    high resolution one.
    
    >>> ind = Match_triangles(high_x, high_y, high_z, low_x, low_y, low_z)
    >>> n_lowres = ind.shape
    """
    code = """
    double dot, new_dot;
    
    for (int i=0; i<n_highres; i++) {
        dot = 0.;
        new_dot = 0.;
        for (int j=0; j<n_lowres; j++) {
            new_dot = high_x(i)*low_x(j) + high_y(i)*low_y(j) + high_z(i)*low_z(j);
            if (new_dot > dot) {
                dot = new_dot;
                ind(i) = j;
            }
        }
    }
    """
    n_highres = high_x.size
    n_lowres = low_x.size
    ind = np.zeros(n_highres, dtype='int')
    get_assoc = scipy.weave.inline(code, ['n_highres','n_lowres','ind', 'high_x', 'low_x', 'high_y', 'low_y', 'high_z', 'low_z'], type_converters=scipy.weave.converters.blitz, compiler='gcc', libraries=['m'], verbose=2, force=0)
    return ind

def Match_subtriangles(inds_highres, inds_lowres):
    """Match_subtriangles(inds_highres, inds_lowres)
    
    Given a list of match of triangles at one resolution (say 4 to 3)
    and another at a higher resolution (say 5 to 4), will match the 
    higher resolution with the base resolution (5 to 3).
    
    >>> ind = Match_subtriangles(inds_highres, inds_lowres)
    >>> inds_highres.shape = ind.shape
    """
    code = """
    int tmp_ind;
    
    for (int i=0; i<n_highres; i++) {
        tmp_ind = inds_highres(i);
        ind(i) = inds_lowres(tmp_ind);
    }
    """
    n_highres = inds_highres.size
    n_lowres = inds_lowres.size
    ind = np.zeros(n_highres, dtype='int')
    get_assoc = scipy.weave.inline(code, ['n_highres','n_lowres','ind', 'inds_highres', 'inds_lowres'], type_converters=scipy.weave.converters.blitz, compiler='gcc', libraries=['m'], verbose=2, force=0)
    return ind


