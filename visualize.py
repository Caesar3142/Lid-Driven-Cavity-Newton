"""
Visualization utilities for lid-driven cavity flow results
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import csv
import os
from solver import LidDrivenCavitySolver
from discrete_loss import DiscreteLossNS


def plot_velocity_field(solution, nx, ny, Lx=1.0, Ly=1.0, save_path=None):
    """
    Plot velocity field with streamlines
    
    Args:
        solution: Dictionary with 'u', 'v' fields
        nx, ny: Grid dimensions
        Lx, Ly: Domain dimensions
        save_path: Optional path to save figure
    """
    u = solution['u']
    v = solution['v']
    
    # Create grid
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    
    # Compute velocity magnitude
    speed = np.sqrt(u**2 + v**2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Velocity magnitude contour
    im1 = ax1.contourf(X, Y, speed, levels=20, cmap=cm.viridis)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Velocity Magnitude')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, label='|u|')
    
    # Streamlines
    ax2.streamplot(X, Y, u, v, density=1.5, color=speed, cmap=cm.viridis, linewidth=1.5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Streamlines')
    ax2.set_aspect('equal')
    ax2.set_xlim(0, Lx)
    ax2.set_ylim(0, Ly)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_pressure_field(solution, nx, ny, Lx=1.0, Ly=1.0, save_path=None):
    """
    Plot pressure field
    
    Args:
        solution: Dictionary with 'p' field
        nx, ny: Grid dimensions
        Lx, Ly: Domain dimensions
        save_path: Optional path to save figure
    """
    p = solution['p']
    
    # Create grid
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    
    # Pressure contour
    im = ax.contourf(X, Y, p, levels=20, cmap=cm.seismic)
    ax.contour(X, Y, p, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Pressure Field')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Pressure')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_centerline_velocity(solution, nx, ny, Lx=1.0, Ly=1.0, save_path=None):
    """
    Plot velocity profiles at centerlines (benchmark comparison)
    
    Args:
        solution: Dictionary with 'u', 'v' fields
        nx, ny: Grid dimensions
        Lx, Ly: Domain dimensions
        save_path: Optional path to save figure
    """
    u = solution['u']
    v = solution['v']
    
    # Create grid
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    
    # Centerline indices
    mid_x = nx // 2
    mid_y = ny // 2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # u-velocity along vertical centerline (x = Lx/2)
    ax1.plot(u[:, mid_x], y, 'b-', linewidth=2, label='u-velocity')
    ax1.set_xlabel('u-velocity')
    ax1.set_ylabel('y')
    ax1.set_title('u-velocity along vertical centerline (x = Lx/2)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # v-velocity along horizontal centerline (y = Ly/2)
    ax2.plot(x, v[mid_y, :], 'r-', linewidth=2, label='v-velocity')
    ax2.set_xlabel('x')
    ax2.set_ylabel('v-velocity')
    ax2.set_title('v-velocity along horizontal centerline (y = Ly/2)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_convergence_history(solution, save_path=None):
    """
    Plot loss convergence history
    
    Args:
        solution: Dictionary with 'loss_history' and 'loss_components_history'
        save_path: Optional path to save figure
    """
    loss_history = solution['loss_history']
    loss_components = solution['loss_components_history']
    
    iterations = np.arange(1, len(loss_history) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Total loss
    ax1.semilogy(iterations, loss_history, 'b-', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Total Loss (log scale)')
    ax1.set_title('Total Loss Convergence')
    ax1.grid(True, alpha=0.3)
    
    # Loss components
    momentum_x = [comp['momentum_x'] for comp in loss_components]
    momentum_y = [comp['momentum_y'] for comp in loss_components]
    continuity = [comp['continuity'] for comp in loss_components]
    
    ax2.semilogy(iterations, momentum_x, 'r-', linewidth=2, label='Momentum X')
    ax2.semilogy(iterations, momentum_y, 'g-', linewidth=2, label='Momentum Y')
    ax2.semilogy(iterations, continuity, 'b-', linewidth=2, label='Continuity')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss Component (log scale)')
    ax2.set_title('Loss Components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def export_centerline_velocity_csv(solution, nx, ny, Lx=1.0, Ly=1.0, save_dir=None):
    """
    Export centerline velocity profiles to CSV files
    
    Args:
        solution: Dictionary with 'u', 'v' fields
        nx, ny: Grid dimensions
        Lx, Ly: Domain dimensions
        save_dir: Optional directory to save CSV files
    """
    u = solution['u']
    v = solution['v']
    
    # Create grid
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    
    # Centerline indices
    mid_x = nx // 2
    mid_y = ny // 2
    
    # u-velocity along vertical centerline (x = Lx/2)
    u_centerline = u[:, mid_x]
    
    # v-velocity along horizontal centerline (y = Ly/2)
    v_centerline = v[mid_y, :]
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Export u-velocity along vertical centerline
    u_csv_path = f"{save_dir}/u_velocity_centerline.csv" if save_dir else "u_velocity_centerline.csv"
    with open(u_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['y', 'u_velocity'])  # Header
        for i in range(len(y)):
            writer.writerow([y[i], u_centerline[i]])
    print(f"u-velocity centerline data saved to {u_csv_path}")
    
    # Export v-velocity along horizontal centerline
    v_csv_path = f"{save_dir}/v_velocity_centerline.csv" if save_dir else "v_velocity_centerline.csv"
    with open(v_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'v_velocity'])  # Header
        for i in range(len(x)):
            writer.writerow([x[i], v_centerline[i]])
    print(f"v-velocity centerline data saved to {v_csv_path}")


def plot_all_results(solution, nx, ny, Lx=1.0, Ly=1.0, save_dir=None):
    """
    Generate all visualization plots
    
    Args:
        solution: Dictionary with solution fields and loss history
        nx, ny: Grid dimensions
        Lx, Ly: Domain dimensions
        save_dir: Optional directory to save figures
    """
    print("Generating visualizations...")
    
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
    
    # Velocity field
    plot_velocity_field(solution, nx, ny, Lx, Ly, 
                       save_path=f"{save_dir}/velocity_field.png" if save_dir else None)
    
    # Pressure field
    plot_pressure_field(solution, nx, ny, Lx, Ly,
                       save_path=f"{save_dir}/pressure_field.png" if save_dir else None)
    
    # Centerline velocities
    plot_centerline_velocity(solution, nx, ny, Lx, Ly,
                            save_path=f"{save_dir}/centerline_velocity.png" if save_dir else None)
    
    # Convergence history
    plot_convergence_history(solution,
                            save_path=f"{save_dir}/convergence.png" if save_dir else None)
    
    # Export centerline velocity CSV files
    export_centerline_velocity_csv(solution, nx, ny, Lx, Ly, save_dir=save_dir)
    
    print("Visualization complete!")

