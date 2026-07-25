# ray-casting-sedimentation

A 3D ray casting algorithm for classifying simulation grid nodes as fluid, solid, or boundary — applied to sedimentation of spherical and spheroidal particles in viscoelastic fluids. Includes validation through volume convergence studies and real-time 2D/3D visualizations.

---

## Table of Contents

- [Background](#background)
- [Algorithm](#algorithm)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Validation](#validation)
- [Visualizations](#visualizations)
- [Results](#results)
- [Next Steps](#next-steps)

---

## Background

In grid-based numerical simulations of particle sedimentation, every node in the simulation domain must be classified as one of three types:

| Node Type | Description | Physics |
|-----------|-------------|---------|
| **FLUID** | Outside all particles | Fluid flow equations apply |
| **SOLID** | Inside a particle | Treated as rigid body |
| **BOUNDARY** | On particle surface | No-slip condition applies |

As particles settle under gravity through a viscoelastic fluid, they move through the fixed grid — requiring reclassification of nodes at every timestep. This module implements the ray casting algorithm to perform this classification efficiently and correctly for both spherical and spheroidal particles.

**Why viscoelastic fluids?**
Unlike simple Newtonian fluids (water, air), viscoelastic fluids have an elastic memory — they remember past deformation. This makes particle motion complex and unpredictable, requiring full numerical simulation. Examples include polymer solutions, biological fluids, and mud.

**Why ray casting over distance-based methods?**
Distance-based methods (e.g. checking if a node is within radius R of the particle center) only work for spheres. Ray casting works for any closed surface — spheres, spheroids, or arbitrary shapes — making it the universal solution for particle sedimentation simulations.

---

## Algorithm

### Core Principle — Jordan Curve Theorem

> A ray shot from any point in any direction will cross the boundary of a closed surface an **odd** number of times if the point is **inside**, and an **even** number of times if it is **outside**.

### Ray-Surface Intersection

A ray from node O in direction D is defined as:
```
P(t) = O + t·D,   D = (1, 0, 0)   [always +X direction]
```

**For a sphere** (center C, radius R), substituting P(t) into the sphere equation gives a quadratic:
```
at² + bt + c = 0

a = D·D
b = 2(V·D),   V = O - C
c = V·V - R²
```

**For a spheroid** (center C, semi-axes a, b, c), the same substitution gives:
```
At² + Bt + C_coef = 0

A      = (Dx/a)² + (Dy/b)² + (Dz/c)²
B      = 2(VxDx/a² + VyDy/b² + VzDz/c²)
C_coef = F(O) - 1,   F(O) = (Vx/a)² + (Vy/b)² + (Vz/c)²
```

### Classification Rules

```
Δ = b² - 4ac (discriminant)

Δ < 0              → ray misses surface entirely → FLUID
t1 < 0, t2 > 0     → node is inside particle    → SOLID
both t > 0          → node is outside particle   → FLUID
both t < 0          → particle behind node       → FLUID
```

### Boundary Detection — Two Methods

**Method 1: Implicit Function F(P)** *(shape-specific)*
```
Sphere:    |distance to surface| < epsilon  →  BOUNDARY
Spheroid:  |F(O) - 1| < epsilon            →  BOUNDARY

Caveat: non-uniform shell thickness — same epsilon corresponds 
to different true distances near poles vs equator of a spheroid
```

**Method 2: Neighbour Pass** *(geometry-independent, preferred)*
```
Any FLUID node with at least one SOLID face-neighbour (±X, ±Y, ±Z)
is reclassified as BOUNDARY.

Advantages:
✓ Exactly one node thick everywhere — perfectly uniform shell
✓ Works for any shape — no distance formula needed
✓ No epsilon tuning required
```

---

## Project Structure

```
ray-casting-sedimentation/
│
├── ray_casting.py       ← Core algorithm (all classify functions)
├── validation.py        ← Proves correctness (convergence, shell check, visuals)
├── animation.py         ← Application demo (2D animations, 3D snapshots)
│
└── validation_plots/
    ├── sphere_convergence.png
    ├── spheroid_convergence.png
    └── cross_section_overlay.png
```

---

## Installation

**Requirements:**
```
Python 3.x
NumPy
Matplotlib
```

**Install dependencies:**
```bash
pip install numpy matplotlib
```

**Clone the repository:**
```bash
git clone https://github.com/[your-username]/ray-casting-sedimentation.git
cd ray-casting-sedimentation
```

---

## Usage

### Running the Core Algorithm

```python
from ray_casting import (
    classify_node_sphere,
    classify_node_spheroid,
    classify_node_multi_sphere,
    classify_fluid_solid_spheroid,
    neighbour_pass,
    grid_summary,
    FLUID, SOLID, BOUNDARY
)
import numpy as np

# Single sphere
C = np.array([5.0, 5.0, 5.0])
R = 2.0
O = np.array([3.0, 5.0, 5.0])

result = classify_node_sphere(O, C, R, epsilon=0.1)
print(result)   # BOUNDARY

# Single spheroid (prolate, stretched in Z)
a, b, c = 2.0, 2.0, 3.5
result = classify_node_spheroid(O, C, a, b, c, epsilon=0.1)

# Multiple spheres
spheres = [(np.array([3.0, 5.0, 5.0]), 1.5),
           (np.array([7.0, 5.0, 5.0]), 1.0)]
result = classify_node_multi_sphere(O, spheres, epsilon=0.1)
```

### Building a Full 3D Grid with Neighbour Pass

```python
import numpy as np
from ray_casting import classify_fluid_solid_spheroid, neighbour_pass, grid_summary

# Grid setup
grid_size = 10
spacing   = 0.5
x_vals    = np.arange(0, grid_size, spacing)
y_vals    = np.arange(0, grid_size, spacing)
z_vals    = np.arange(0, grid_size, spacing)

C       = np.array([5.0, 5.0, 5.0])
a, b, c = 2.0, 2.0, 3.5

# Build grid
grid = {}
for x in x_vals:
    for y in y_vals:
        for z in z_vals:
            O   = np.array([x, y, z])
            key = (round(x,4), round(y,4), round(z,4))
            grid[key] = classify_fluid_solid_spheroid(O, C, a, b, c)

# Apply neighbour pass for boundary detection
grid = neighbour_pass(grid, spacing)
grid_summary(grid)
```

---

## Validation

Run the full validation suite:
```bash
python validation.py
```

### Section 1 — Sanity Checks
Tests known geometric positions against expected classifications for both sphere and spheroid:
```
[PASS] node at center          expected=SOLID    got=SOLID
[PASS] node far outside        expected=FLUID    got=FLUID
[PASS] node on surface (X)     expected=BOUNDARY got=BOUNDARY
[PASS] node on surface (Z)     expected=BOUNDARY got=BOUNDARY
[PASS] node behind spheroid    expected=FLUID    got=FLUID
```

### Section 2 & 3 — Volume Convergence Study

Compares numerical volume (solid_count × spacing³) against exact analytical volume at multiple grid spacings. Error decreases as grid is refined — confirming correct convergence behaviour.

| Spacing | Solid Nodes | Numerical V | Error % |
|---------|-------------|-------------|---------|
| 1.00 | ... | ... | ~15% |
| 0.75 | ... | ... | ~10% |
| 0.50 | ... | ... | ~6%  |
| 0.25 | ... | ... | ~3%  |

**Sphere exact volume:** (4/3)πR³

**Spheroid exact volume:** (4/3)πabc

### Section 4 — Boundary Shell Thickness
Verifies every boundary node has at least one solid neighbour — confirming the neighbour pass produces a uniform, exactly-one-node-thick shell everywhere.

### Section 5 — Visual Cross-Section Overlay
2D slice through the center of the 3D grid overlaid with exact analytical surface outlines. Green boundary nodes should sit exactly on the dashed analytical line.

---

## Visualizations

Run the animation and 3D visualization demo:
```bash
python animation.py
```

### Section 1 — Single Sphere 2D Settling Animation
Real-time node reclassification as a sphere settles under gravity. Sphere rests at y = R (surface touching floor).

### Section 2 — Multi-Sphere 2D Animation
Two spheres of different radii (R1=1.2, R2=0.8) settling simultaneously. Velocities derived from **Stokes' Law:**
```
v = (2/9) × (ρ_particle - ρ_fluid) × g × R² / μ
```
Larger particle falls faster — physically correct. Raw Stokes velocities scaled to fit abstract grid (non-dimensionalization).

### Section 3 — 3D Spheroid Static Snapshots
Three timesteps shown side by side: start, mid-settling, and settled position. Uses neighbour pass boundary detection. Fluid nodes hidden for clarity; boundary plotted with transparency so solid core is visible.

---

## Results

```
Grid size: 10×10×10, spacing=0.5 → 8000 total nodes
Spheroid: center=(5,5,5), a=b=2.0, c=3.5 (prolate)

Node classification:
  Fluid    : 7253   (90.7%)
  Solid    :  457   ( 5.7%)
  Boundary :  290   ( 3.6%)
  Total    : 8000   ✓
```

---

## Next Steps

- [ ] Compute outward unit normals at boundary nodes using finite difference gradient of phase field
- [ ] Convert Python codebase to C
- [ ] Integrate with existing viscoelastic fluid solver (Oldroyd-B model)
- [ ] Study sedimentation of spheroidal particles in viscoelastic fluid
- [ ] Extend to handle deforming particle geometries

---

---

## License

MIT License
