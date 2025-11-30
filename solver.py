"""
Main Solver for Lid-Driven Cavity Flow using Discrete Loss Optimization
"""

import numpy as np
import torch
import torch.optim as optim
from torch.nn import Parameter
from discrete_loss import DiscreteLossNS
from boundary_conditions import LidDrivenCavityBC


class LidDrivenCavitySolver:
    """
    Solver for lid-driven cavity flow using discrete loss optimization
    """
    
    def __init__(self, nx=41, ny=41, Lx=1.0, Ly=1.0, Re=100, rho=1.0, U_lid=1.0):
        """
        Initialize solver
        
        Args:
            nx: Number of grid points in x-direction
            ny: Number of grid points in y-direction
            Lx: Domain length in x-direction
            Ly: Domain length in y-direction
            Re: Reynolds number
            rho: Fluid density
            U_lid: Lid velocity
        """
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.Re = Re
        self.rho = rho
        self.U_lid = U_lid
        
        # Initialize discrete loss framework
        self.discrete_loss = DiscreteLossNS(nx, ny, Lx, Ly, Re, rho)
        
        # Initialize boundary conditions
        self.bc = LidDrivenCavityBC(nx, ny, Lx, Ly, U_lid)
        
        # Initialize fields as parameters
        self.u = Parameter(torch.zeros(ny, nx))
        self.v = Parameter(torch.zeros(ny, nx))
        self.p = Parameter(torch.zeros(ny, nx))
        
    def reset_fields(self):
        """Reset velocity and pressure fields to zero"""
        with torch.no_grad():
            self.u.data.zero_()
            self.v.data.zero_()
            self.p.data.zero_()
    
    def solve(self, max_iter=10000, lr=0.01, tol=1e-6, verbose=True, log_interval=100):
        """
        Solve the lid-driven cavity flow problem
        
        Args:
            max_iter: Maximum number of iterations
            lr: Learning rate
            tol: Convergence tolerance
            verbose: Print progress
            log_interval: Print loss every N iterations
            
        Returns:
            Dictionary with solution fields and loss history
        """
        # Reset fields
        self.reset_fields()
        
        # Set up optimizer
        optimizer = optim.Adam([self.u, self.v, self.p], lr=lr)
        
        # Loss history
        loss_history = []
        loss_components_history = []
        
        for iteration in range(max_iter):
            optimizer.zero_grad()
            
            # Apply boundary conditions
            u_bc, v_bc, p_bc = self.bc.apply_boundary_conditions(self.u, self.v, self.p)
            
            # Compute discrete loss
            loss, loss_components = self.discrete_loss.compute_discrete_loss(u_bc, v_bc, p_bc)
            
            # Backward pass
            loss.backward()
            
            # Zero out gradients at boundary points (boundary values are fixed)
            with torch.no_grad():
                interior_mask = self.bc.get_interior_mask()
                boundary_mask = self.bc.get_boundary_mask()
                
                # Only update interior points
                self.u.grad[boundary_mask] = 0.0
                self.v.grad[boundary_mask] = 0.0
                # Pressure can vary at boundaries except reference point
                # self.p.grad[0, 0] = 0.0  # Reference pressure point
            
            optimizer.step()
            
            # Apply boundary conditions again after update
            with torch.no_grad():
                u_bc, v_bc, p_bc = self.bc.apply_boundary_conditions(self.u, self.v, self.p)
                self.u.data = u_bc.data
                self.v.data = v_bc.data
                # Set pressure reference point (fix pressure at one point to remove singularity)
                self.p.data = self.p.data - self.p.data[0, 0]  # Normalize so p[0,0] = 0
            
            # Record loss
            loss_history.append(loss.item())
            loss_components_history.append(loss_components)
            
            # Check convergence
            if loss.item() < tol:
                if verbose:
                    print(f"Converged at iteration {iteration+1} with loss = {loss.item():.2e}")
                break
            
            # Print progress
            if verbose and (iteration + 1) % log_interval == 0:
                print(f"Iteration {iteration+1}/{max_iter}: Loss = {loss.item():.2e}, "
                      f"Momentum_x = {loss_components['momentum_x']:.2e}, "
                      f"Momentum_y = {loss_components['momentum_y']:.2e}, "
                      f"Continuity = {loss_components['continuity']:.2e}")
        
        # Final solution
        u_final, v_final, p_final = self.bc.apply_boundary_conditions(self.u, self.v, self.p)
        
        return {
            'u': u_final.detach().numpy(),
            'v': v_final.detach().numpy(),
            'p': p_final.detach().numpy(),
            'loss_history': loss_history,
            'loss_components_history': loss_components_history,
            'iterations': iteration + 1
        }


if __name__ == "__main__":
    # Example usage
    print("Solving lid-driven cavity flow problem...")
    
    # Create solver
    solver = LidDrivenCavitySolver(nx=41, ny=41, Re=100)
    
    # Solve
    solution = solver.solve(max_iter=5000, lr=0.01, verbose=True)
    
    print(f"\nSolution completed in {solution['iterations']} iterations")
    print(f"Final loss: {solution['loss_history'][-1]:.2e}")

