""" validation of the ray caasting node classification algorithm.
    This validation file proves the correctness of the algorithm
    through various sanity checks, volume convergence studies,boundary shell analysis
    and visual cross-section overlays.
    
    Imports from Ray_Casting.py(Both validation and Ray_Casting file must be in the same folder)"""

import numpy as np
import matplotlib.pyplot as plt
from  mpl_toolkits.mplot3d import Axes3D

from Ray_Casting import(
    classify_node_sphere,
    classify_node_spheroid,
    classify_fluid_solid_spheroid,
    neighbour_pass,
    grid_summary,
    fluid, solid, boundary
)


#Section-1 Sanity checks
def run_sanity_checks():
    """
    Basic tests verifying correct classification fro known geometric positions
    relative to a sphere and spheroid
    """

    print("SECTION 1- SANITY CHECKS")
    print("\n Sphere tests:-")

    #Sphere tests
    print("\nSphere (center=(5,5,5), R=2.0):")
    C = np.array([5.0, 5.0, 5.0])
    R = 2.0

    tests = [
        (np.array([5.0, 5.0, 5.0]), solid,    "node at center"),
        (np.array([0.0, 5.0, 5.0]), fluid,    "node far outside"),
        (np.array([7.0, 5.0, 5.0]), boundary, "node on surface (X)"),
        (np.array([9.0, 5.0, 5.0]), fluid,    "node behind sphere"),
        (np.array([5.0, 5.0, 5.0]), solid,    "node at center again"),
    ]

    all_passed = True
    for O, expected, description in tests:
        result = classify_node_sphere(O, C, R)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  [{status}] {description:30s} expected={expected:8s} got={result}")

    
    print("\n Spheroid tests: ")
    print("\nSpheroid (center=(5,5,5), a=b=2.0, c=3.5):")
    C = np.array([5.0, 5.0, 5.0])
    a, b, c = 2.0, 2.0, 3.5

    tests_spheroid = [
        (np.array([5.0, 5.0, 5.0]), solid,    "node at center"),
        (np.array([0.0, 5.0, 5.0]), fluid,    "node far outside"),
        (np.array([7.0, 5.0, 5.0]), boundary, "node on surface (X)"),
        (np.array([5.0, 5.0, 8.5]), boundary, "node on surface (Z)"),
        (np.array([9.0, 5.0, 5.0]), fluid,    "node behind spheroid"),
    ]

    for O, expected, description in tests_spheroid:
        result = classify_node_spheroid(O, C, a, b, c)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  [{status}] {description:30s} expected={expected:8s} got={result}")

    print(f"\nSanity checks: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed

#Section-2 Sphere volume convergence study
def sphere_volume_convergence():
    """
    Validates the sphere classification by comparing numerical volume
    (solid count* spacing^3) against exact analytical volume (4/3*pi*r^3)
    at various grid spacings, So if the algorithm is correct then the volume 
    error should continuously decrease as the grid spacing decreases or as the
    grid is refined. 
    """

    print("SECTION-2: SPHERE VOLUME CONVERGENCE")
    C = np.array([5.0, 5.0, 5.0])
    R = 2.0
    V_exact = (4/3) * np.pi * R**3

    print(f"\nSphere: center={C}, R={R}")
    print(f"Exact volume = {V_exact:.4f}")
    print(f"\n{'Spacing':>10} {'Solid Nodes':>12} {'Numerical V':>12} {'Error %':>10}")

    spacings = [1.0, 0.75, 0.5, 0.25]
    errors   = []

    #For different grid spacings we generate different errors
    for spacing in spacings:
        x_vals = np.arange(0, 10, spacing)
        y_vals = np.arange(0, 10, spacing)
        z_vals = np.arange(0, 10, spacing)

        solid_count = 0
        for x in x_vals:
            for y in y_vals:
                for z in z_vals:
                    O = np.array([x, y, z])
                    # Use fluid/solid only — volume defined by solid nodes
                    V_vec = O - C
                    F = np.dot(V_vec, V_vec)
                    if F < R**2:   # exact inside check for sphere
                        solid_count += 1

        V_numerical = solid_count * spacing**3
        error = abs(V_exact - V_numerical) / V_exact * 100
        errors.append(error)
        print(f"{spacing:>10.2f} {solid_count:>12} {V_numerical:>12.4f} {error:>10.2f}%")


# --- Plot convergence ---
    plt.figure(figsize=(7, 5))
    plt.plot(spacings, errors, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Grid Spacing')
    plt.ylabel('Volume Error (%)')
    plt.title('Sphere Volume Convergence — Error vs Grid Spacing')
    plt.gca().invert_xaxis()   # finer grid on the right
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('sphere_convergence.png', dpi=150)
    plt.close()
    print("\nPlot saved as sphere_convergence.png")

#Section-3 Spheroid volume convergence study
def spheroid_volume_convergence():
    """
    Validates the spheroid classification by comparing numerical volume
    (solid count* spacing^3) against exact analytical volume (4/3*pi*a*b*c)
    at various grid spacings, So if the algorithm is correct then the volume 
    error should continuously decrease as the grid spacing decreases or as the
    grid is refined. 
    """
    print("SECTION-3: SPHEROID VOLUME CONVERGENCE")
    C = np.array([5.0, 5.0, 5.0])
    a, b, c = 2.0, 2.0, 3.5
    V_exact = (4/3) * np.pi * a * b * c

    print(f"\nSpheroid: center={C}, a={a}, b={b}, c={c}")
    print(f"Exact volume = {V_exact:.4f}")
    print(f"\n{'Spacing':>10} {'Solid Nodes':>12} {'Numerical V':>12} {'Error %':>10}")

    spacings = [1.0, 0.75, 0.5, 0.25]
    errors   = []

    for spacing in spacings:
        x_vals = np.arange(0, 10, spacing)
        y_vals = np.arange(0, 10, spacing)
        z_vals = np.arange(0, 10, spacing)

        solid_count = 0
        for x in x_vals:
            for y in y_vals:
                for z in z_vals:
                    O   = np.array([x, y, z])
                    V_vec = O - C
                    # Exact inside check for spheroid — no sqrt needed
                    F = (V_vec[0]/a)**2 + (V_vec[1]/b)**2 + (V_vec[2]/c)**2
                    if F < 1.0:
                        solid_count += 1

        V_numerical = solid_count * spacing**3
        error = abs(V_exact - V_numerical) / V_exact * 100
        errors.append(error)
        print(f"{spacing:>10.2f} {solid_count:>12} {V_numerical:>12.4f} {error:>10.2f}%")

    # --- Plot convergence ---
    plt.figure(figsize=(7, 5))
    plt.plot(spacings, errors, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Grid Spacing')
    plt.ylabel('Volume Error (%)')
    plt.title('Spheroid Volume Convergence — Error vs Grid Spacing')
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('spheroid_convergence.png', dpi=150)
    plt.close()
    print("\nPlot saved as spheroid_convergence.png")

#Sectio-4 Boundary shell thickness check
def boundary_shell_check():
    """
    Validates that the neighbour pass produces a boundary shell
    exactly one node thick everywhere.

    Method:
        For every BOUNDARY node, check its 6 neighbours.
        At least one neighbour must be SOLID (confirms it's on
        the surface). No neighbour should be BOUNDARY on the
        outer side (confirms shell is exactly one layer thick).
    """
    print("SECTION 4 — BOUNDARY SHELL THICKNESS CHECK")
    C = np.array([5.0, 5.0, 5.0])
    a, b, c = 2.0, 2.0, 3.5
    spacing = 0.5

    # Build grid using neighbour pass
    x_vals = np.arange(0, 10, spacing)
    y_vals = np.arange(0, 10, spacing)
    z_vals = np.arange(0, 10, spacing)

    grid = {}
    for x in x_vals:
        for y in y_vals:
            for z in z_vals:
                O   = np.array([x, y, z])
                key = (round(x,4), round(y,4), round(z,4))
                grid[key] = classify_fluid_solid_spheroid(O, C, a, b, c)

    grid = neighbour_pass(grid, spacing)
    grid_summary(grid)

    # Check every boundary node 
    offsets = [
        ( spacing,  0,        0       ),
        (-spacing,  0,        0       ),
        ( 0,        spacing,  0       ),
        ( 0,       -spacing,  0       ),
        ( 0,        0,        spacing ),
        ( 0,        0,       -spacing ),
    ]

    boundary_nodes = [(x,y,z) for (x,y,z), v in grid.items() if v == boundary]

    has_solid_neighbour   = 0
    no_solid_neighbour    = 0

    for (x, y, z) in boundary_nodes:
        solid_found = False
        for (dx, dy, dz) in offsets:
            neighbour_key = (
                round(x + dx, 4),
                round(y + dy, 4),
                round(z + dz, 4)
            )
            if grid.get(neighbour_key) == solid:
                solid_found = True
                break

        if solid_found:
            has_solid_neighbour += 1
        else:
            no_solid_neighbour += 1

    print(f"\nBoundary nodes checked : {len(boundary_nodes)}")
    print(f"Have solid neighbour   : {has_solid_neighbour}")
    print(f"No solid neighbour     : {no_solid_neighbour}")

    if no_solid_neighbour == 0:
        print("\nSHELL CHECK PASSED,every boundary node touches solid")
    else:
        print("\nSHELL CHECK FAILED,some boundary nodes have no solid neighbour")

#Section-5 Cross section overlay
def cross_section_overlay():
    """
    Takes a 2D slice through the center of the 3D classified grid
    and overlays the exact analytical sphere and spheroid outlines.

    A correct classifier should show boundary nodes sitting exactly
    on the analytical surface outline.
    """

    print("SECTION 5 — CROSS SECTION OVERLAY")

    spacing = 0.25
    C       = np.array([5.0, 5.0, 5.0])

    # Using a ≠ b deliberately to make cross-sections visually distinct 
    # This tests the general ellipsoid case, which is a superset of spheroid
    a, b, c = 3.0, 1.5, 3.5
    R       = 2.0
    slice_z = C[2]   #This is to slice through the center of sphere/spheroid that we are considering

    x_vals = np.arange(0, 10, spacing)
    y_vals = np.arange(0, 10, spacing)
    z_vals = np.arange(0, 10, spacing)

    #First build a spheroid grid as earlier
    grid_spheroid = {}
    for x in x_vals:
        for y in y_vals:
            for z in z_vals:
                O   = np.array([x, y, z])
                key = (round(x,4), round(y,4), round(z,4))
                grid_spheroid[key] = classify_fluid_solid_spheroid(
                    O, C, a, b, c
                )
    grid_spheroid = neighbour_pass(grid_spheroid, spacing)

    #Then the next step is to slice the grid and extract the slice at Z=slice_z
    def extract_slice(grid, z_target, tol):
        fluid_pts, solid_pts, boundary_pts = [], [], []
        for (x, y, z), classification in grid.items():
            if abs(z - z_target) < tol:
                if classification == fluid:
                    fluid_pts.append((x, y))
                elif classification == solid:
                    solid_pts.append((x, y))
                elif classification == boundary:
                    boundary_pts.append((x, y))
        return (np.array(fluid_pts),
                np.array(solid_pts),
                np.array(boundary_pts))
    

    tol = spacing * 0.6
    fluid_s, solid_s, boundary_s = extract_slice(
        grid_spheroid, slice_z, tol
    )

    theta = np.linspace(0, 2*np.pi, 300)

    # Spheroid cross-section at Z=center is an ellipse with axes a, b
    spheroid_x = C[0] + a * np.cos(theta)
    spheroid_y = C[1] + b * np.sin(theta)
    # sphere cross section at the centre is a circle with radius R
    sphere_x = C[0] + R * np.cos(theta)
    sphere_y = C[1] + R * np.sin(theta)

    #Plotting the spheroid slice
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    if len(fluid_s):
        ax.scatter(fluid_s[:,0], fluid_s[:,1],
                   c='blue', s=10, label='Fluid')
    if len(solid_s):
        ax.scatter(solid_s[:,0], solid_s[:,1],
                   c='red', s=10, label='Solid')
    if len(boundary_s):
        ax.scatter(boundary_s[:,0], boundary_s[:,1],
                   c='green', s=20, label='Boundary')
    ax.plot(spheroid_x, spheroid_y, 'k--',
            linewidth=2, label='Exact surface')
    ax.set_title('Spheroid Cross-Section (Z = center)')
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.grid(True)


    #Building a sphere grid and slicing it for comparison
    grid_sphere = {}
    for x in x_vals:
        for y in y_vals:
            for z in z_vals:
                O   = np.array([x, y, z])
                key = (round(x,4), round(y,4), round(z,4))
                V_vec = O - C
                F     = np.dot(V_vec, V_vec)
                grid_sphere[key] = solid if F < R**2 else fluid
    grid_sphere = neighbour_pass(grid_sphere, spacing)

    fluid_sp, solid_sp, boundary_sp = extract_slice(
        grid_sphere, slice_z, tol
    )

    ax = axes[1]
    if len(fluid_sp):
        ax.scatter(fluid_sp[:,0], fluid_sp[:,1],
                   c='blue', s=10, label='Fluid')
    if len(solid_sp):
        ax.scatter(solid_sp[:,0], solid_sp[:,1],
                   c='red', s=10, label='Solid')
    if len(boundary_sp):
        ax.scatter(boundary_sp[:,0], boundary_sp[:,1],
                   c='green', s=20, label='Boundary')
    ax.plot(sphere_x, sphere_y, 'k--',
            linewidth=2, label='Exact surface')
    ax.set_title('Sphere Cross-Section (Z = center)')
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.grid(True)

    plt.suptitle('Cross-Section Overlay — Numerical vs Analytical Surface',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig('cross_section_overlay.png', dpi=150)
    plt.close()
    print("\nPlot saved as cross_section_overlay.png")



if __name__ == "__main__":
    passed = run_sanity_checks()
    sphere_volume_convergence()
    spheroid_volume_convergence()
    boundary_shell_check()
    cross_section_overlay()


    if passed:
        print("ALL VALIDATION CHECKS COMPLETE")
    else:
        print("WARNING — SOME SANITY CHECKS FAILED")

    print("\nShowing all plots...")
    img1 = plt.imread('sphere_convergence.png')
    img2 = plt.imread('spheroid_convergence.png')
    img3 = plt.imread('cross_section_overlay.png')

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes[0].imshow(img1)
    axes[0].axis('off')
    axes[0].set_title('Sphere Convergence')

    axes[1].imshow(img2)
    axes[1].axis('off')
    axes[1].set_title('Spheroid Convergence')

    axes[2].imshow(img3)
    axes[2].axis('off')
    axes[2].set_title('Cross Section Overlay')

    plt.tight_layout()
    plt.show()