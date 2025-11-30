"""
Example usage of the lid-driven cavity flow solver

This script demonstrates how to use the discrete loss optimization framework
to solve the lid-driven cavity flow problem with different parameters.
"""

import numpy as np
from solver import LidDrivenCavitySolver
from visualize import plot_velocity_field, plot_pressure_field, plot_convergence_history


def example_basic():
    """Basic example: Solve at Re=100"""
    print("=" * 60)
    print("Example 1: Basic lid-driven cavity flow at Re=100")
    print("=" * 60)
    
    solver = LidDrivenCavitySolver(nx=41, ny=41, Re=100)
    solution = solver.solve(max_iter=5000, lr=0.01, verbose=True, log_interval=500)
    
    print(f"\nSolved in {solution['iterations']} iterations")
    print(f"Final loss: {solution['loss_history'][-1]:.2e}")
    
    # Visualize
    plot_velocity_field(solution, 41, 41)
    plot_pressure_field(solution, 41, 41)
    plot_convergence_history(solution)


def example_different_reynolds():
    """Example: Solve at different Reynolds numbers"""
    print("=" * 60)
    print("Example 2: Comparing different Reynolds numbers")
    print("=" * 60)
    
    reynolds_numbers = [100, 400, 1000]
    
    for Re in reynolds_numbers:
        print(f"\nSolving for Re = {Re}...")
        solver = LidDrivenCavitySolver(nx=51, ny=51, Re=Re)
        solution = solver.solve(max_iter=5000, lr=0.005, verbose=False)
        
        print(f"  Re = {Re}: {solution['iterations']} iterations, "
              f"final loss = {solution['loss_history'][-1]:.2e}")


def example_fine_grid():
    """Example: Solve on a finer grid"""
    print("=" * 60)
    print("Example 3: Fine grid resolution (81x81)")
    print("=" * 60)
    
    solver = LidDrivenCavitySolver(nx=81, ny=81, Re=100)
    
    # Use smaller learning rate for finer grids
    solution = solver.solve(max_iter=3000, lr=0.005, verbose=True, log_interval=300)
    
    print(f"\nSolved in {solution['iterations']} iterations")
    print(f"Final loss: {solution['loss_history'][-1]:.2e}")


def example_custom_parameters():
    """Example: Custom domain and parameters"""
    print("=" * 60)
    print("Example 4: Custom parameters")
    print("=" * 60)
    
    # Rectangular domain
    solver = LidDrivenCavitySolver(
        nx=61, 
        ny=41, 
        Lx=1.5, 
        Ly=1.0, 
        Re=200,
        U_lid=2.0
    )
    
    solution = solver.solve(max_iter=5000, lr=0.008, verbose=True, log_interval=500)
    
    print(f"\nSolved in {solution['iterations']} iterations")
    
    # Visualize with custom domain size
    plot_velocity_field(solution, 61, 41, Lx=1.5, Ly=1.0)


if __name__ == "__main__":
    # Run examples
    print("\nLid-Driven Cavity Flow Solver - Examples")
    print("=" * 60)
    
    # Uncomment the example you want to run:
    
    example_basic()
    # example_different_reynolds()
    # example_fine_grid()
    # example_custom_parameters()
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)

