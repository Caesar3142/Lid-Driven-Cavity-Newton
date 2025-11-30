# Lid-Driven Cavity Flow Solver using Discrete Loss Optimization

This project implements a solver for the lid-driven cavity flow problem using an **Optimizing a Discrete Loss (ODIL)** framework. The approach discretizes the Navier-Stokes equations using finite differences and optimizes the discrete residuals directly.

## Problem Description

The lid-driven cavity flow is a classic benchmark problem in computational fluid dynamics. It consists of:
- A square cavity (1×1 domain)
- Three stationary walls (bottom, left, right) with **free-slip** boundary conditions
- One moving wall (top lid) moving with constant velocity
- Incompressible, viscous flow governed by Navier-Stokes equations

### Governing Equations

For steady-state flow, we solve:

**Momentum equations:**
```
(u·∇)u = -∇p/ρ + ν∇²u
(u·∇)v = -∇p/ρ + ν∇²v
```

**Continuity equation:**
```
∇·u = 0
```

where:
- `u`, `v` are the x and y velocity components
- `p` is the pressure
- `ρ` is the fluid density
- `ν` is the kinematic viscosity

The Reynolds number is defined as: `Re = UL/ν`, where `U` is the lid velocity and `L` is the cavity length.

## Methodology: Discrete Loss Optimization

Instead of using traditional iterative solvers (like SIMPLE or PISO), this framework:

1. **Discretizes** the PDEs using finite differences on a collocated grid
2. **Computes residuals** at interior grid points for:
   - X-momentum equation
   - Y-momentum equation
   - Continuity equation
3. **Optimizes** the sum of squared residuals using gradient-based optimization (Adam optimizer)
4. **Enforces boundary conditions** at each iteration

The discrete loss is:
```
L = Σ(R_x² + R_y² + R_cont²)
```

where `R_x`, `R_y`, and `R_cont` are the residuals of the momentum and continuity equations.

## Installation

1. Clone or download this repository

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
# venv\Scripts\activate   # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note:** On macOS with Homebrew Python, using a virtual environment is strongly recommended to avoid conflicts with system-managed packages.

## Usage

**Important:** Make sure to activate your virtual environment first:
```bash
source venv/bin/activate  # On macOS/Linux
```

### Basic Usage

Run the main script:
```bash
python main.py
```

This will:
- Solve the lid-driven cavity flow at Re=100
- Display convergence progress and timing information
- Generate visualization plots
- Export centerline velocity data to CSV files

### Custom Parameters

You can modify parameters in `main.py` or use the solver directly:

```python
from solver import LidDrivenCavitySolver

# Create solver with custom parameters
solver = LidDrivenCavitySolver(
    nx=200,      # Grid points in x-direction
    ny=200,      # Grid points in y-direction
    Re=100,     # Reynolds number
    Lx=1.0,     # Domain length in x
    Ly=1.0,     # Domain length in y
    U_lid=1.0   # Lid velocity
)

# Solve
solution = solver.solve(
    max_iter=10000,  # Maximum iterations
    lr=0.01,         # Learning rate
    tol=1e-6,        # Convergence tolerance
    verbose=True     # Print progress
)

# Access results
u = solution['u']  # x-velocity field
v = solution['v']  # y-velocity field
p = solution['p']  # pressure field
loss_history = solution['loss_history']
```

### Visualization

The package includes visualization utilities:

```python
from visualize import plot_all_results

# Generate all plots
plot_all_results(solution, nx, ny, save_dir="results")
```

Available visualization functions:
- `plot_velocity_field()` - Velocity magnitude and streamlines
- `plot_pressure_field()` - Pressure contours
- `plot_centerline_velocity()` - Velocity profiles at centerlines
- `plot_convergence_history()` - Loss convergence curves
- `export_centerline_velocity_csv()` - Export centerline velocity data to CSV files

### Data Export

The solver automatically exports centerline velocity profiles to CSV files:
- `results/u_velocity_centerline.csv` - u-velocity along vertical centerline (x = Lx/2)
- `results/v_velocity_centerline.csv` - v-velocity along horizontal centerline (y = Ly/2)

These CSV files are useful for:
- Comparing with benchmark data (e.g., Ghia et al. 1982)
- Post-processing and analysis
- Validation studies

## Project Structure

```
LDC-ODIL/
├── discrete_loss.py      # Discrete loss framework for Navier-Stokes
├── boundary_conditions.py # Boundary condition implementation
├── solver.py             # Main solver class
├── visualize.py          # Visualization utilities
├── main.py               # Example usage script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Key Components

### 1. `DiscreteLossNS` (`discrete_loss.py`)
Computes discrete residuals for:
- X-momentum equation using central differences
- Y-momentum equation using central differences  
- Continuity equation

### 2. `LidDrivenCavityBC` (`boundary_conditions.py`)
Handles boundary conditions with **free-slip** walls:
- Top wall: `u = U_lid`, `v = 0` (moving lid)
- Bottom wall: `v = 0` (normal), `∂u/∂y = 0` (free-slip)
- Left wall: `u = 0` (normal), `∂v/∂x = 0` (free-slip)
- Right wall: `u = 0` (normal), `∂v/∂x = 0` (free-slip)

Free-slip means:
- Normal component of velocity = 0 (no flow through wall)
- Tangential component has zero gradient (no shear stress, fluid can slide along wall)

### 3. `LidDrivenCavitySolver` (`solver.py`)
Main solver that:
- Initializes velocity and pressure fields
- Optimizes discrete loss using Adam optimizer
- Enforces boundary conditions at each iteration
- Monitors convergence

## Results

The solver generates:
- Velocity fields (magnitude and streamlines)
- Pressure field contours
- Centerline velocity profiles (for benchmark comparison)
- Convergence history plots
- CSV files with centerline velocity data
- Timing information (solving time, visualization time, total runtime)

## Notes

- The pressure field is normalized to have zero reference at the bottom-left corner
- The solver uses a collocated grid (all variables at same grid points)
- Boundary conditions are enforced explicitly at each iteration
- **Free-slip boundary conditions** are used for the stationary walls (bottom, left, right)
- The top wall maintains a moving lid with specified velocity
- For stability, the learning rate may need adjustment for different Reynolds numbers
- Timing information is displayed showing solving time, visualization time, and total runtime

## References

- Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of computational physics*, 48(3), 387-411.
- LeVeque, R. J. (2007). *Finite difference methods for ordinary and partial differential equations*. SIAM.

## License

This code is provided for educational and research purposes.

