"""
animation.py

Visualizations and animations demonstrating the ray casting
node classification algorithm applied to particle sedimentation.

Section 1 :- Single sphere 2D settling animation
Section 2 :- Multi sphere 2D settling animation (Stokes' Law velocities)
Section 3 :- 3D static snapshots of spheroid classification

Imports from Ray_Casting.py must be in the same folder.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Ray_Casting import (
    classify_node_sphere,
    classify_node_multi_sphere,
    classify_fluid_solid_spheroid,
    neighbour_pass,
    fluid,solid,boundary
)

#Global declaration of the grid details since it will be acccessed by all the sections of the code
grid_size = 10
spacing   = 0.25
x_values  = np.arange(0, grid_size, spacing)
y_values  = np.arange(0, grid_size, spacing)
D         = np.array([1.0, 0.0, 0.0])

#Section 1 Single sphere animation
def single_sphere_animation():
    R        = 1.5
    sphere_x = 5.0
    start_y  = 9.0
    end_y    = R        # rests with surface touching floor

    num_frames     = 60
    y_positions    = np.linspace(start_y, end_y, num_frames)

    fig, ax = plt.subplots(figsize=(7, 7))

    def update(frame):
        ax.clear()

        C = np.array([sphere_x, y_positions[frame], 0.0])

        fluid_x,    fluid_y    = [], []
        solid_x,    solid_y    = [], []
        boundary_x, boundary_y = [], []

        for x in x_values:
            for y in y_values:
                O      = np.array([x, y, 0.0])
                result = classify_node_sphere(O, C, R, epsilon=0.15)

                if result == fluid:
                    fluid_x.append(x)
                    fluid_y.append(y)
                elif result == solid:
                    solid_x.append(x)
                    solid_y.append(y)
                elif result == boundary:
                    boundary_x.append(x)
                    boundary_y.append(y)

        ax.scatter(fluid_x,    fluid_y,    c='blue',  marker='.',  s=15)
        ax.scatter(solid_x,    solid_y,    c='red',   marker='.',  s=15)
        ax.scatter(boundary_x, boundary_y, c='green', marker='.',  s=20)

        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(C[0] + R*np.cos(theta),
                C[1] + R*np.sin(theta), 'k--', linewidth=1)

        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_title(f'Single Sphere Settling Frame {frame+1}/{num_frames}')
        ax.set_aspect('equal')
        ax.grid(True)

    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=50, repeat=True
    )
    plt.show()

#Section 2 Multi sphere animation
def multi_sphere_animation():
    """
    Animates two spheres of different radii settling under gravity.
    Velocities derived from Stokes' Law, larger particle falls faster.
    Particles stop when surface touches floor (end_y = R).
    """

    #Parameters for stokes law
    g     = 9.8
    rho_p = 1200.0   # particle density
    rho_f = 1000.0   # fluid density
    mu    = 0.5      # fluid viscosity

    R1 = 1.2
    R2 = 0.8

    # Raw Stokes velocities
    v1_raw = (2/9) * (rho_p - rho_f) * g * R1**2 / mu
    v2_raw = (2/9) * (rho_p - rho_f) * g * R2**2 / mu

    # Scale to fit grid
    scale  = 0.1 / max(v1_raw, v2_raw)
    v1     = v1_raw * scale
    v2     = v2_raw * scale

    print(f"Stokes velocities (scaled): v1={v1:.4f}, v2={v2:.4f} units/frame")
    print(f"Velocity ratio v1/v2 = {v1/v2:.3f} (physically derived from R1/R2)")

    # --- Sphere paths ---
    sphere1_x       = 3.5
    sphere2_x       = 7.0
    sphere1_start_y = 9.0
    sphere2_start_y = 7.0
    sphere1_end_y   = R1
    sphere2_end_y   = R2

    frames_1   = int((sphere1_start_y - sphere1_end_y) / v1)
    frames_2   = int((sphere2_start_y - sphere2_end_y) / v2)
    num_frames = max(frames_1, frames_2) + 5

    sphere1_y = np.maximum(
        sphere1_start_y - v1 * np.arange(num_frames),
        sphere1_end_y
    )
    sphere2_y = np.maximum(
        sphere2_start_y - v2 * np.arange(num_frames),
        sphere2_end_y
    )

    fig, ax = plt.subplots(figsize=(7, 7))

    def update(frame):
        ax.clear()

        C1      = np.array([sphere1_x, sphere1_y[frame], 0.0])
        C2      = np.array([sphere2_x, sphere2_y[frame], 0.0])
        spheres = [(C1, R1), (C2, R2)]

        fluid_x,    fluid_y    = [], []
        solid_x,    solid_y    = [], []
        boundary_x, boundary_y = [], []

        for x in x_values:
            for y in y_values:
                O      = np.array([x, y, 0.0])
                result = classify_node_multi_sphere(O, spheres, epsilon=0.15)

                if result == fluid:
                    fluid_x.append(x)
                    fluid_y.append(y)
                elif result == solid:
                    solid_x.append(x)
                    solid_y.append(y)
                elif result == boundary:
                    boundary_x.append(x)
                    boundary_y.append(y)

        ax.scatter(fluid_x,    fluid_y,    c='blue',  marker='.',  s=15)
        ax.scatter(solid_x,    solid_y,    c='red',   marker='s',  s=15)
        ax.scatter(boundary_x, boundary_y, c='green', marker='^',  s=20)

        theta = np.linspace(0, 2*np.pi, 100)
        for (C, R) in spheres:
            ax.plot(C[0] + R*np.cos(theta),
                    C[1] + R*np.sin(theta), 'k--', linewidth=1)

        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_title(
            f'Multi-Sphere Sedimentation Frame {frame+1}/{num_frames}\n'
            f'R1={R1} (v={v1:.3f})   R2={R2} (v={v2:.3f})'
        )
        ax.set_aspect('equal')
        ax.grid(True)

    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=50, repeat=True
    )
    plt.show()

#Section 3 Spheroid snapshot classification(Since 3d animation takes a lot of rendering time)
def spheroid_3d_snapshots():
    """
    Shows three static 3D snapshots of a spheroid at different
    positions, top, middle and bottom simulating sedimentation.
    Uses neighbour pass for boundary detection.
    """
    a, b, c = 2.0, 2.0, 3.5
    R_floor = c        # rests with tip touching floor

    # Three positions start, middle, settled
    positions = [
        np.array([5.0, 5.0, 8.0]),   # near top
        np.array([5.0, 5.0, 5.5]),   # middle
        np.array([5.0, 5.0, R_floor]),  # resting on floor
    ]
    titles = ['Start', 'Mid-Settling', 'Settled']

    z_values = np.arange(0, grid_size, spacing)

    fig = plt.figure(figsize=(18, 6))

    for idx, C in enumerate(positions):

        grid = {}
        for x in x_values:
            for y in y_values:
                for z in z_values:
                    O   = np.array([x, y, z])
                    key = (round(x,4), round(y,4), round(z,4))
                    grid[key] = classify_fluid_solid_spheroid(
                        O, C, a, b, c
                    )
        grid = neighbour_pass(grid, spacing)

        # Collect solid and boundary points
        solid_pts    = []
        boundary_pts = []
        for (x, y, z), classification in grid.items():
            if classification == solid:
                solid_pts.append((x, y, z))
            elif classification == boundary:
                boundary_pts.append((x, y, z))

        solid_pts    = np.array(solid_pts)
        boundary_pts = np.array(boundary_pts)

        # Plot
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')

        if len(solid_pts):
            ax.scatter(solid_pts[:,0], solid_pts[:,1], solid_pts[:,2],
                       c='red', marker='o', s=10,
                       depthshade=False, label='Solid')

        if len(boundary_pts):
            ax.scatter(boundary_pts[:,0], boundary_pts[:,1], boundary_pts[:,2],
                       c='green', marker='^', s=10, alpha=0.1,
                       depthshade=False, label='Boundary')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{titles[idx]}\nCenter Z = {C[2]}')
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_zlim(0, grid_size)
        ax.legend(fontsize=7)

    plt.suptitle(
        '3D Spheroid Sedimentation, Node Classification at Three Timesteps',
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig('spheroid_3d_snapshots.png', dpi=150)
    plt.show()
    print("Plot saved as spheroid_3d_snapshots.png")


if __name__ == "__main__":
    print("Running Section 1 — Single Sphere Animation...")
    single_sphere_animation()

    print("Running Section 2 — Multi Sphere Animation...")
    multi_sphere_animation()

    print("Running Section 3 — 3D Snapshots...")
    spheroid_3d_snapshots()