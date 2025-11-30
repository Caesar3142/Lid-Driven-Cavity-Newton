"""
Discrete Loss Framework for Navier-Stokes Equations
Optimizing a Discrete Loss (ODIL) approach for solving fluid dynamics problems
"""

import numpy as np
import torch
import torch.nn as nn


class DiscreteLossNS:
    """
    Discrete loss formulation for incompressible Navier-Stokes equations:
    
    Continuity: ∇·u = 0
    Momentum: ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + f
    
    For steady-state lid-driven cavity, we solve:
    -∇p/ρ + ν∇²u + (u·∇)u = 0
    ∇·u = 0
    """
    
    def __init__(self, nx, ny, Lx=1.0, Ly=1.0, Re=100, rho=1.0):
        """
        Initialize discrete loss framework
        
        Args:
            nx: Number of grid points in x-direction
            ny: Number of grid points in y-direction
            Lx: Domain length in x-direction
            Ly: Domain length in y-direction
            Re: Reynolds number (Re = U*L/ν, where U is lid velocity, L is cavity length)
            rho: Fluid density
        """
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.Re = Re
        self.rho = rho
        
        # Grid spacing
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        
        # Velocity scale (lid velocity)
        self.U_lid = 1.0
        # Kinematic viscosity from Reynolds number
        self.nu = self.U_lid * max(Lx, Ly) / Re
        
        # Create grid
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Grid indices
        self.i = np.arange(nx)
        self.j = np.arange(ny)
        
    def compute_momentum_residual_x(self, u, v, p):
        """
        Compute x-momentum residual:
        (u·∇)u + ∂p/∂x/ρ - ν∇²u = 0
        
        Using central differences on collocated grid
        """
        # Interior points only (1:-1 slices)
        u_int = u[1:-1, 1:-1]  # Interior u values
        
        # Convective terms (u·∇)u
        du_dx = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * self.dx)
        du_dy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * self.dy)
        
        # Interpolate v to u-locations for cross-derivative (average of 4 corners)
        v_at_u = (v[1:-1, 1:-1] + v[2:, 1:-1] + v[1:-1, 2:] + v[2:, 2:]) / 4
        
        conv_x = u_int * du_dx + v_at_u * du_dy
        
        # Pressure gradient
        dp_dx = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * self.dx)
        pressure_term = dp_dx / self.rho
        
        # Viscous term (Laplacian)
        d2u_dx2 = (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / (self.dx**2)
        d2u_dy2 = (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / (self.dy**2)
        viscous_term = self.nu * (d2u_dx2 + d2u_dy2)
        
        residual_x = conv_x + pressure_term - viscous_term
        return residual_x
    
    def compute_momentum_residual_y(self, u, v, p):
        """
        Compute y-momentum residual:
        (u·∇)v + ∂p/∂y/ρ - ν∇²v = 0
        
        Using central differences on collocated grid
        """
        # Interior points only (1:-1 slices)
        v_int = v[1:-1, 1:-1]  # Interior v values
        
        # Convective terms (u·∇)v
        dv_dx = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * self.dx)
        dv_dy = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * self.dy)
        
        # Interpolate u to v-locations for cross-derivative (average of 4 corners)
        u_at_v = (u[1:-1, 1:-1] + u[1:-1, 2:] + u[2:, 1:-1] + u[2:, 2:]) / 4
        
        conv_y = u_at_v * dv_dx + v_int * dv_dy
        
        # Pressure gradient
        dp_dy = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * self.dy)
        pressure_term = dp_dy / self.rho
        
        # Viscous term (Laplacian)
        d2v_dx2 = (v[1:-1, 2:] - 2*v[1:-1, 1:-1] + v[1:-1, :-2]) / (self.dx**2)
        d2v_dy2 = (v[2:, 1:-1] - 2*v[1:-1, 1:-1] + v[:-2, 1:-1]) / (self.dy**2)
        viscous_term = self.nu * (d2v_dx2 + d2v_dy2)
        
        residual_y = conv_y + pressure_term - viscous_term
        return residual_y
    
    def compute_continuity_residual(self, u, v):
        """
        Compute continuity residual:
        ∇·u = 0
        """
        # Velocity gradients
        du_dx = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * self.dx)
        dv_dy = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * self.dy)
        
        residual_continuity = du_dx + dv_dy
        return residual_continuity
    
    def compute_discrete_loss(self, u, v, p):
        """
        Compute total discrete loss as sum of squared residuals
        """
        # Momentum residuals
        res_x = self.compute_momentum_residual_x(u, v, p)
        res_y = self.compute_momentum_residual_y(u, v, p)
        
        # Continuity residual
        res_cont = self.compute_continuity_residual(u, v)
        
        # Total loss (sum of squared residuals)
        loss = (torch.sum(res_x**2) + torch.sum(res_y**2) + torch.sum(res_cont**2))
        
        return loss, {
            'momentum_x': torch.sum(res_x**2).item(),
            'momentum_y': torch.sum(res_y**2).item(),
            'continuity': torch.sum(res_cont**2).item()
        }

