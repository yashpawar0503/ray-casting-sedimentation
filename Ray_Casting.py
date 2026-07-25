""" ray_Casting.py

This file contains the code for the Ray casting algorithm for classifying grid nodes as FLUID, SOLID or BOUNDARY in a partical sedimentation
simulation.

Application:
    Sedimentation of spherical and spheroidal particles in a
    viscoelastic fluid. At every timestep, this module classifies
    every node in the grid based on its position relative to
    the particles.

Node Types:
    FLUID    - Outside all particles. Fluid flow equations apply.
    SOLID    - Inside a particle. Treated as rigid body.
    BOUNDARY - On particle surface. No-slip condition applies.

Author: Yash Pawar
Date: 24/06/2026

""" 
#Import all the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Node classfication labels to make our lives easier later on
fluid="FLUID"
solid="SOLID"
boundary="BOUNDARY"

#Ray direction (always +X for all classification calls)
ray_direction=np.array([1.0,0.0,0.0])


#Classifying a single grid node based on the position of the node and the sedimenting sphere
def classify_node_sphere(O,C,R,epsilon=0.1):
    """ Classification of a single grid node based on the relative position of the sphere and the node
    
        Parameters:
            O:node position, np.array shape(3,)
            C:sphere center, np.array shape(3,)
            R:sphere radius, float
            epsilon: boundary tolerance(recommended 0.5*grid spacing), float

        Returns:
            str- "FLUID", "SOLID" or"BOUNDARY"

    """

    D=ray_direction

    #step-1 is to check the distance of the node from the sphere and check for boundary classification
    distance=np.linalg.norm(O-C)-R
    if abs(distance)<epsilon:
        return boundary
    
    #Step-2 is to calculate coefficients
    V=O-C
    a=np.dot(D,D)
    b=2*np.dot(V,D)
    c=np.dot(V,V)-R**2

    #Step-3 is to calculate discriminant
    discriminant=b**2-4*a*c
    if discriminant<0:
        return fluid
    
    t1=(-b-np.sqrt(discriminant))/(2*a)
    t2=(-b+np.sqrt(discriminant))/(2*a)

    if t1<0 and t2>0:
        return solid
    
    else:
        return fluid
    
def classify_node_spheroid(O,C,a,b,c,epsilon=0.1):
    """classify a single grid node relative to a spheroid
       Parameters:
         O:node position, np.array shape(3,)
         C:sphere center, np.array shape(3,)
         a:semi axis along X, float
         b:semi axis along Y, float
         c:semi axis along Z, float
         epsilon: boundary tolerance(recommended 0.5*grid spacing), float

        Returns:
            str- "FLUID", "SOLID" or"BOUNDARY"

        Note: True spheroid: a=b≠c
        Prolate: c>a (stretched in the z direction)
        Oblate: c<a (flattened in the z direction)
        sphere is the special case of spheroid where a=b=c

        Boundary warning:
        We use implicit function F(P) for boundary detection. F is not true geometric distance
        same epsilon gives largertrue distance near the poles hence more nodes qualify as the boundary
        near the poles (long axis) ans smaller near the equator(short axis). We use neighbour_pass()for uniform
        geometry independent boundary detection.

    """

    D=ray_direction

    #Step-1 is to calculate the implicit function F and boundary check

    V=O-C
    F=(V[0]/a)**2 + (V[1]/b)**2 + (V[2]/c)**2
    C_coef=F-1

    if abs(C_coef)<epsilon:
        return boundary
    
    # Step 2 is to caculate Quadratic coefficients
    A = (D[0]/a)**2 + (D[1]/b)**2 + (D[2]/c)**2
    B = 2 * ((V[0]*D[0])/a**2 + (V[1]*D[1])/b**2 + (V[2]*D[2])/c**2)

    #Step-3 is to calculate the discriminant
    discriminant=B**2-4*A*C_coef
    if discriminant<0:
        return fluid
    t1 = (-B - np.sqrt(discriminant)) / (2*A)
    t2 = (-B + np.sqrt(discriminant)) / (2*A)

    if t1 < 0 and t2 > 0:
        return solid
    else:
        return fluid
    
def classify_node_multi_sphere(O,spheres,epsilon=0.1):
    """ Now here we classify a single grid node based on the existence of multiple spheres in the grid

        Parameters:
          O:node position, np.array shape(3,)
          spheres : list of (C, R) tuples - [(center1, radius1), ...]
          epsilon: boundary threshold, float
        
        Returns:
          str - "FLUID", "SOLID", or "BOUNDARY"
        
        Logic:
            BOUNDARY if node is on surface of ANY sphere (checked first)
            SOLID    if node is inside ANY sphere
            FLUID    only if it passes all checks for all spheres     
    """

    #Step-1 is the boundary check for all spheres
    for (C,R) in spheres:
        distance= np.linalg.norm(O-C)-R
        if abs(distance)<epsilon:
            return boundary

    #Step-2 is the solid check for all the spheres   
    for(C,R) in spheres:
        V=O-C
        D= ray_direction
        a=np.dot(D,D)
        b = 2 * np.dot(V, D)
        c = np.dot(V, V) - R**2

        discriminant = b**2 - 4*a*c
        if discriminant >= 0:
             t1 = (-b - np.sqrt(discriminant)) / (2*a)
             t2 = (-b + np.sqrt(discriminant)) / (2*a)
            
             if t1 < 0 and t2 > 0:
                return solid

    return fluid

def classify_fluid_solid_spheroid(O,C,a,b,c):
    """Here we classify nodes as fluid and solid only, No boundary check
       Parameters:
            O:node position, np.array shape(3,)
            C:sphere center, np.array shape(3,)
            a:semi axis along X, float
            b:semi axis along Y, float
            c:semi axis along Z, float

       Returns:
            str - "FLUID" or "SOLID" only (never "BOUNDARY")
       
            
        Here we intentionally dont have the epsilon parameter since we are only classifying nodes
        as fluid and solid we will use another function called neighbour_pass() after this function
        has classified all the nodes
    """

    D=ray_direction
    V      = O - C
    F      = (V[0]/a)**2 + (V[1]/b)**2 + (V[2]/c)**2
    C_coef = F - 1

    A = (D[0]/a)**2 + (D[1]/b)**2 + (D[2]/c)**2
    B = 2 * ((V[0]*D[0])/a**2 + (V[1]*D[1])/b**2 + (V[2]*D[2])/c**2)

    discriminant = B**2 - 4*A*C_coef

    if discriminant < 0:
        return fluid

    t1 = (-B - np.sqrt(discriminant)) / (2*A)
    t2 = (-B + np.sqrt(discriminant)) / (2*A)

    if t1 < 0 and t2 > 0:
        return solid
    else:
        return fluid

def neighbour_pass(grid, spacing):
    """
    Reclassify fluid nodes adjacent to solid nodes as boundary nodes.
    No epsilon needed here and this doesnt depend on the geometry.

    Parameters:
        grid    : dict, {(x, y, z): classification}
                  keys must be rounded to 4 decimal places
        spacing : float, grid spacing

    Returns:
        dict, updated grid with boundary nodes reclassified

    Method:
        For every FLUID node, check its 6 face-neighbours (±X, ±Y, ±Z).
        If any neighbour is SOLID then reclassify as BOUNDARY.

    Advantages over F(P) epsilon method:
        - Exactly one node layer thick everywhere
        - No epsilon tuning required
        - Geometry independent, same code for any geometry
    """
    offsets = [
        ( spacing,  0,        0       ),   # +X
        (-spacing,  0,        0       ),   # -X
        ( 0,        spacing,  0       ),   # +Y
        ( 0,       -spacing,  0       ),   # -Y
        ( 0,        0,        spacing ),   # +Z
        ( 0,        0,       -spacing ),   # -Z
    ]

    # Collect nodes to reclassify first
    to_reclassify = []

    for (x, y, z), classification in grid.items():
        if classification == fluid:
            for (dx, dy, dz) in offsets:
                neighbour_key = (
                    round(x + dx, 4),
                    round(y + dy, 4),
                    round(z + dz, 4)
                )
                if grid.get(neighbour_key) == solid:
                    to_reclassify.append((x, y, z))
                    break

    # Apply reclassification after iteration is complete
    for key in to_reclassify:
        grid[key] = boundary

    return grid

def grid_summary(grid):
    """
    Print total no of node counts in the grid.

    Parameters:
        grid dictionary
    """
    fluid_cnt    = sum(1 for v in grid.values() if v == fluid)
    solid_cnt    = sum(1 for v in grid.values() if v == solid)
    boundary_cnt = sum(1 for v in grid.values() if v == boundary)
    total    = len(grid)

    print(f"Grid Summary:")
    print(f"  Total    : {total}")
    print(f"  Fluid    : {fluid_cnt}     ({100*fluid_cnt/total:.1f}%)")
    print(f"  Solid    : {solid_cnt}     ({100*solid_cnt/total:.1f}%)")
    print(f"  Boundary : {boundary_cnt}     ({100*boundary_cnt/total:.1f}%)")
    print(f"  Check    : {fluid_cnt+solid_cnt+boundary_cnt} == {total} → {'OK' if fluid_cnt+solid_cnt+boundary_cnt==total else 'ERROR'}")



if __name__ == "__main__":

    print("=" * 40)
    print("Testing ray_casting.py")
    print("=" * 40)

    # --- Test 1: Sphere classifier ---
    print("\n--- Sphere Tests ---")
    C_sphere = np.array([5.0, 5.0, 5.0])
    R = 2.0

    tests_sphere = [
        (np.array([5.0, 5.0, 5.0]), solid,    "center of sphere"),
        (np.array([0.0, 5.0, 5.0]), fluid,    "far outside sphere"),
        (np.array([7.0, 5.0, 5.0]), boundary, "on sphere surface"),
        (np.array([9.0, 5.0, 5.0]), fluid,    "sphere behind node"),
    ]

    for O, expected, description in tests_sphere:
        result = classify_node_sphere(O, C_sphere, R)
        status = "OK" if result == expected else "FAIL"
        print(f"  [{status}] {description}: expected {expected}, got {result}")

    # --- Test 2: Spheroid classifier ---
    print("\n--- Spheroid Tests ---")
    C_spheroid = np.array([5.0, 5.0, 5.0])
    a, b, c = 2.0, 2.0, 3.5

    tests_spheroid = [
        (np.array([5.0, 5.0, 5.0]), solid,    "center of spheroid"),
        (np.array([0.0, 5.0, 5.0]), fluid,    "far outside spheroid"),
        (np.array([7.0, 5.0, 5.0]), boundary, "on spheroid surface X"),
        (np.array([5.0, 5.0, 8.5]), boundary, "on spheroid surface Z"),
        (np.array([9.0, 5.0, 5.0]), fluid,    "spheroid behind node"),
    ]

    for O, expected, description in tests_spheroid:
        result = classify_node_spheroid(O, C_spheroid, a, b, c)
        status = "OK" if result == expected else "FAIL"
        print(f"  [{status}] {description}: expected {expected}, got {result}")

    # --- Test 3: Grid summary ---
    print("\n--- Grid Summary Test ---")
    grid = {}
    spacing = 0.5
    x_vals = np.arange(0, 10, spacing)
    y_vals = np.arange(0, 10, spacing)
    z_vals = np.arange(0, 10, spacing)

    for x in x_vals:
        for y in y_vals:
            for z in z_vals:
                O   = np.array([x, y, z])
                key = (round(x,4), round(y,4), round(z,4))
                grid[key] = classify_fluid_solid_spheroid(O, C_spheroid, a, b, c)

    grid = neighbour_pass(grid, spacing)
    grid_summary(grid)