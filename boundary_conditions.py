"""
Boundary Conditions for Lid-Driven Cavity Flow
"""

import numpy as np
import torch


class LidDrivenCavityBC:
    """
    Boundary conditions for lid-driven cavity with free-slip walls:
    - Top wall (y=Ly): u = U_lid, v = 0 (moving lid)
    - Bottom wall (y=0): v = 0 (normal), ∂u/∂y = 0 (free-slip)
    - Left wall (x=0): u = 0 (normal), ∂v/∂x = 0 (free-slip)
    - Right wall (x=Lx): u = 0 (normal), ∂v/∂x = 0 (free-slip)
    
    Free-slip means:
    - Normal component of velocity = 0 (no flow through wall)
    - Tangential component has zero gradient (no shear stress)
    """
    
    def __init__(self, nx, ny, Lx=1.0, Ly=1.0, U_lid=1.0):
        """
        Initialize boundary conditions
        
        Args:
            nx: Number of grid points in x-direction
            ny: Number of grid points in y-direction
            Lx: Domain length in x-direction
            Ly: Domain length in y-direction
            U_lid: Lid velocity (top wall)
        """
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.U_lid = U_lid
        
    def apply_boundary_conditions(self, u, v, p):
        """
        Apply boundary conditions to velocity and pressure fields with free-slip walls
        
        Args:
            u: x-velocity field (ny x nx)
            v: y-velocity field (ny x nx)
            p: pressure field (ny x nx)
            
        Returns:
            u, v, p with boundary conditions applied
        """
        # Copy to avoid modifying original
        u_bc = u.clone() if isinstance(u, torch.Tensor) else u.copy()
        v_bc = v.clone() if isinstance(v, torch.Tensor) else v.copy()
        p_bc = p.clone() if isinstance(p, torch.Tensor) else p.copy()
        
        # Top wall (y = Ly, j = ny-1): moving lid
        u_bc[-1, :] = self.U_lid  # u = U_lid (tangential velocity)
        v_bc[-1, :] = 0.0  # v = 0 (normal component)
        
        # Bottom wall (y = 0, j = 0): free-slip
        v_bc[0, :] = 0.0  # v = 0 (normal component)
        # Free-slip: ∂u/∂y = 0, so u[0,:] = u[1,:] (zero gradient)
        # But exclude corners which will be handled separately
        u_bc[0, 1:-1] = u_bc[1, 1:-1]  # Tangential velocity has zero gradient
        
        # Left wall (x = 0, i = 0): free-slip
        u_bc[:, 0] = 0.0  # u = 0 (normal component)
        # Free-slip: ∂v/∂x = 0, so v[:,0] = v[:,1] (zero gradient)
        # But exclude corners which will be handled separately
        v_bc[1:-1, 0] = v_bc[1:-1, 1]  # Tangential velocity has zero gradient
        
        # Right wall (x = Lx, i = nx-1): free-slip
        u_bc[:, -1] = 0.0  # u = 0 (normal component)
        # Free-slip: ∂v/∂x = 0, so v[:,-1] = v[:,-2] (zero gradient)
        # But exclude corners which will be handled separately
        v_bc[1:-1, -1] = v_bc[1:-1, -2]  # Tangential velocity has zero gradient
        
        # Corner points: satisfy normal components from both walls
        # Bottom-left corner (x=0, y=0)
        u_bc[0, 0] = 0.0  # Normal component from left wall
        v_bc[0, 0] = 0.0  # Normal component from bottom wall
        
        # Bottom-right corner (x=Lx, y=0)
        u_bc[0, -1] = 0.0  # Normal component from right wall
        v_bc[0, -1] = 0.0  # Normal component from bottom wall
        
        # Top-left corner (x=0, y=Ly)
        u_bc[-1, 0] = self.U_lid  # From top wall (tangential)
        v_bc[-1, 0] = 0.0  # Normal component
        
        # Top-right corner (x=Lx, y=Ly)
        u_bc[-1, -1] = self.U_lid  # From top wall (tangential)
        v_bc[-1, -1] = 0.0  # Normal component
        
        # Pressure boundary conditions (Neumann: ∂p/∂n = 0 at walls)
        # For simplicity, we'll fix pressure at one point (corner) to remove pressure singularity
        # p_bc[0, 0] = 0.0  # Reference pressure
        
        return u_bc, v_bc, p_bc
    
    def get_boundary_mask(self):
        """
        Get mask for boundary points
        """
        mask = np.ones((self.ny, self.nx), dtype=bool)
        mask[1:-1, 1:-1] = False  # Interior points
        return mask
    
    def get_interior_mask(self):
        """
        Get mask for interior points
        """
        mask = np.zeros((self.ny, self.nx), dtype=bool)
        mask[1:-1, 1:-1] = True  # Interior points
        return mask

