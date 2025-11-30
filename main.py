"""
Main script for solving lid-driven cavity flow using Discrete Loss Optimization
"""

import numpy as np
import time
from solver import LidDrivenCavitySolver
from visualize import plot_all_results, export_centerline_velocity_csv


def main():
    """Main function to solve and visualize lid-driven cavity flow"""
    
    # Record total start time
    total_start_time = time.time()
    
    # Problem parameters
    nx = 200  # Grid points in x-direction
    ny = 200  # Grid points in y-direction
    Re = 100  # Reynolds number
    
    print("=" * 60)
    print("Lid-Driven Cavity Flow Solver")
    print("Using Discrete Loss Optimization Framework")
    print("=" * 60)
    print(f"Grid size: {nx} x {ny}")
    print(f"Reynolds number: Re = {Re}")
    print("-" * 60)
    
    # Create solver
    solver = LidDrivenCavitySolver(nx=nx, ny=ny, Re=Re, Lx=1.0, Ly=1.0, U_lid=1.0)
    
    # Solve
    print("\nSolving...")
    solve_start_time = time.time()
    solution = solver.solve(
        max_iter=10000,
        lr=0.01,
        tol=1e-6,
        verbose=True,
        log_interval=500
    )
    solve_elapsed_time = time.time() - solve_start_time
    
    # Print summary
    print("\n" + "=" * 60)
    print("Solution Summary")
    print("=" * 60)
    print(f"Iterations: {solution['iterations']}")
    print(f"Final loss: {solution['loss_history'][-1]:.2e}")
    print(f"Momentum X residual: {solution['loss_components_history'][-1]['momentum_x']:.2e}")
    print(f"Momentum Y residual: {solution['loss_components_history'][-1]['momentum_y']:.2e}")
    print(f"Continuity residual: {solution['loss_components_history'][-1]['continuity']:.2e}")
    print(f"Solving time: {solve_elapsed_time:.2f} seconds ({solve_elapsed_time/60:.2f} minutes)")
    
    # Compute some flow statistics
    u = solution['u']
    v = solution['v']
    speed = np.sqrt(u**2 + v**2)
    
    print(f"\nFlow Statistics:")
    print(f"Maximum velocity magnitude: {np.max(speed):.4f}")
    print(f"Average velocity magnitude: {np.mean(speed):.4f}")
    print(f"Maximum u-velocity: {np.max(u):.4f}")
    print(f"Maximum v-velocity: {np.max(v):.4f}")
    
    # Visualize results
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)
    
    viz_start_time = time.time()
    plot_all_results(solution, nx, ny, Lx=1.0, Ly=1.0, save_dir="results")
    viz_elapsed_time = time.time() - viz_start_time
    
    # Calculate total elapsed time
    total_elapsed_time = time.time() - total_start_time
    
    print("\n" + "=" * 60)
    print("Timing Summary")
    print("=" * 60)
    print(f"Solving time: {solve_elapsed_time:.2f} seconds ({solve_elapsed_time/60:.2f} minutes)")
    print(f"Visualization time: {viz_elapsed_time:.2f} seconds ({viz_elapsed_time/60:.2f} minutes)")
    print(f"Total running time: {total_elapsed_time:.2f} seconds ({total_elapsed_time/60:.2f} minutes)")
    print("=" * 60)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

