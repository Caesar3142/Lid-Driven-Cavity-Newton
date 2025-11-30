"""
Main Solver for Lid-Driven Cavity Flow using Newton's Method
"""

import numpy as np
import torch
from torch.nn import Parameter
from discrete_loss import DiscreteLossNS
from boundary_conditions import LidDrivenCavityBC


class LidDrivenCavitySolver:
    """
    Solver for lid-driven cavity flow using Newton's method
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
    
    def solve(self, max_iter=100, damping=1.0, tol=1e-6, verbose=True, log_interval=10):
        """
        Solve the lid-driven cavity flow problem using Newton's method
        
        Args:
            max_iter: Maximum number of iterations
            damping: Damping factor for Newton update (0 < damping <= 1)
            tol: Convergence tolerance
            verbose: Print progress
            log_interval: Print loss every N iterations
            
        Returns:
            Dictionary with solution fields and loss history
        """
        # Reset fields
        self.reset_fields()
        
        # Get interior mask for indexing (convert to torch tensor)
        interior_mask_np = self.bc.get_interior_mask()
        interior_mask = torch.tensor(interior_mask_np, dtype=torch.bool, device=self.u.device)
        n_interior = int(torch.sum(interior_mask).item())
        
        # Number of equations: 2 momentum + 1 continuity per interior point
        n_eq = int(3 * n_interior)
        
        # Loss history
        loss_history = []
        loss_components_history = []
        
        for iteration in range(max_iter):
            # Apply boundary conditions - CRITICAL: residuals need boundary values for correct derivatives
            # We need to apply BCs in a way that preserves gradients for interior points
            u_bc, v_bc, p_bc = self.bc.apply_boundary_conditions(self.u, self.v, self.p)
            
            # Create tensors with boundary conditions but preserving gradients for interior
            # The residual computation accesses points near boundaries, so BCs must be correct
            u_with_bc = u_bc.clone()
            v_with_bc = v_bc.clone()
            p_with_bc = p_bc.clone()
            
            # Restore gradient connection for interior points (boundary points don't need gradients)
            u_with_bc[1:-1, 1:-1] = self.u[1:-1, 1:-1]
            v_with_bc[1:-1, 1:-1] = self.v[1:-1, 1:-1]
            p_with_bc[1:-1, 1:-1] = self.p[1:-1, 1:-1]
            
            # Compute residual vector from fields with boundary conditions
            # This ensures derivatives near boundaries use correct boundary values
            residual, loss_components = self.discrete_loss.compute_residual_vector(u_with_bc, v_with_bc, p_with_bc)
            
            # Compute loss for monitoring
            loss = torch.sum(residual**2)
            loss_history.append(loss.item())
            loss_components_history.append(loss_components)
            
            # Check convergence
            residual_norm = torch.norm(residual).item()
            if residual_norm < tol:
                if verbose:
                    print(f"Converged at iteration {iteration+1} with residual norm = {residual_norm:.2e}")
                break
            
            # Print progress
            if verbose and (iteration + 1) % log_interval == 0:
                print(f"Iteration {iteration+1}/{max_iter}: Residual norm = {residual_norm:.2e}, "
                      f"Momentum_x = {loss_components['momentum_x']:.2e}, "
                      f"Momentum_y = {loss_components['momentum_y']:.2e}, "
                      f"Continuity = {loss_components['continuity']:.2e}")
            
            # Compute Jacobian matrix using efficient vectorized approach
            # For large systems, computing full Jacobian is expensive
            # Use a more efficient approach: compute Jacobian-vector products or use approximate methods
            
            # Flatten interior variables for Newton's method
            u_int_flat = self.u[interior_mask].flatten()
            v_int_flat = self.v[interior_mask].flatten()
            p_int_flat = self.p[interior_mask].flatten()
            
            # Combine all variables: [u_int, v_int, p_int]
            x_flat = torch.cat([u_int_flat, v_int_flat, p_int_flat])
            n_vars = len(x_flat)
            
            # For efficiency, use a simplified/approximate Newton method
            # Instead of computing full Jacobian, use gradient of residual norm (Gauss-Newton approximation)
            # or compute Jacobian in batches
            
            if n_eq > 5000:
                # For large systems, use gradient descent on residual norm (simplified Newton)
                # This is much faster but less accurate than full Newton
                if verbose and iteration == 0:
                    print(f"Large system detected ({n_eq} equations). Using simplified Newton method (gradient descent on residual norm).")
                
                # Compute gradient of residual norm with respect to variables
                loss = torch.sum(residual**2)
                grad_u, grad_v, grad_p = torch.autograd.grad(
                    outputs=loss,
                    inputs=[self.u, self.v, self.p],
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True
                )
                
                # Extract interior gradients
                if grad_u is not None:
                    grad_u_flat = grad_u[interior_mask].flatten()
                else:
                    grad_u_flat = torch.zeros(n_interior, dtype=residual.dtype, device=residual.device)
                
                if grad_v is not None:
                    grad_v_flat = grad_v[interior_mask].flatten()
                else:
                    grad_v_flat = torch.zeros(n_interior, dtype=residual.dtype, device=residual.device)
                
                if grad_p is not None:
                    grad_p_flat = grad_p[interior_mask].flatten()
                else:
                    grad_p_flat = torch.zeros(n_interior, dtype=residual.dtype, device=residual.device)
                
                # Use gradient descent direction (simplified Newton)
                delta_x = -torch.cat([grad_u_flat, grad_v_flat, grad_p_flat])
                
            else:
                # For smaller systems, compute full Jacobian (but still expensive)
                if verbose and iteration == 0:
                    print(f"Computing full Jacobian ({n_eq} x {n_vars})...")
                
                jacobian = torch.zeros(n_eq, n_vars, dtype=residual.dtype, device=residual.device)
                
                # Compute Jacobian row by row (this is slow but necessary for exact Newton)
                # Use batching to speed up
                batch_size = min(100, n_eq)  # Process in batches
                for batch_start in range(0, n_eq, batch_size):
                    batch_end = min(batch_start + batch_size, n_eq)
                    batch_indices = range(batch_start, batch_end)
                    
                    for i in batch_indices:
                        residual_i = residual[i]
                        if residual_i.requires_grad:
                            try:
                                grad = torch.autograd.grad(
                                    outputs=residual_i,
                                    inputs=[self.u, self.v, self.p],
                                    retain_graph=(i < n_eq - 1),
                                    create_graph=False,
                                    allow_unused=True
                                )
                                
                                # Flatten gradients and extract interior points
                                if grad[0] is not None:
                                    grad_u_flat = grad[0][interior_mask].flatten()
                                else:
                                    grad_u_flat = torch.zeros(n_interior, dtype=residual.dtype, device=residual.device)
                                
                                if grad[1] is not None:
                                    grad_v_flat = grad[1][interior_mask].flatten()
                                else:
                                    grad_v_flat = torch.zeros(n_interior, dtype=residual.dtype, device=residual.device)
                                
                                if grad[2] is not None:
                                    grad_p_flat = grad[2][interior_mask].flatten()
                                else:
                                    grad_p_flat = torch.zeros(n_interior, dtype=residual.dtype, device=residual.device)
                                
                                # Combine gradients
                                jacobian[i, :] = torch.cat([grad_u_flat, grad_v_flat, grad_p_flat])
                            except RuntimeError:
                                # If gradient computation fails, set row to zero
                                jacobian[i, :] = 0.0
                    
                    if verbose and (batch_start % 1000 == 0 or batch_end == n_eq):
                        print(f"  Computed {batch_end}/{n_eq} rows of Jacobian...")
                
                # Solve Newton system: J * δx = -R
                try:
                    delta_x = torch.linalg.lstsq(jacobian, -residual, rcond=None).solution
                except RuntimeError:
                    # Fallback: use pseudo-inverse for ill-conditioned systems
                    jacobian_pinv = torch.linalg.pinv(jacobian)
                    delta_x = jacobian_pinv @ (-residual)
            
            # Split delta_x into components (already done if using simplified method)
            delta_u_flat = delta_x[:n_interior]
            delta_v_flat = delta_x[n_interior:2*n_interior]
            delta_p_flat = delta_x[2*n_interior:]
            
            # Reshape and apply update with damping
            delta_u = torch.zeros_like(self.u)
            delta_v = torch.zeros_like(self.v)
            delta_p = torch.zeros_like(self.p)
            
            # Reshape flat arrays back to interior mask shape
            # interior_mask is (ny, nx) boolean array, interior points are at [1:-1, 1:-1]
            # The flat arrays are ordered row by row for interior points
            ny_int = self.ny - 2  # Number of interior rows
            nx_int = self.nx - 2  # Number of interior columns
            delta_u[1:-1, 1:-1] = delta_u_flat.reshape(ny_int, nx_int)
            delta_v[1:-1, 1:-1] = delta_v_flat.reshape(ny_int, nx_int)
            delta_p[1:-1, 1:-1] = delta_p_flat.reshape(ny_int, nx_int)
            
            # Update solution with damping
            with torch.no_grad():
                self.u.data += damping * delta_u
                self.v.data += damping * delta_v
                self.p.data += damping * delta_p
                
                # Normalize pressure (fix reference point)
                self.p.data = self.p.data - self.p.data[0, 0]
        
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
    solution = solver.solve(max_iter=100, damping=1.0, verbose=True)
    
    print(f"\nSolution completed in {solution['iterations']} iterations")
    print(f"Final loss: {solution['loss_history'][-1]:.2e}")

