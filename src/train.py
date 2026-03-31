
import torch
from model import PINN
from data import sample_boundary_points, sample_interior_points
from losses import compute_pde_residual, pde_loss, boundary_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

model = PINN().to(device)

interior_points = sample_interior_points(100, device=device)
boundary_points = sample_boundary_points(25, device=device)

print("Interior points shape:", interior_points.shape)
print("Boundary points shape:", boundary_points.shape)

residual = compute_pde_residual(model, interior_points)
loss_pde = pde_loss(model, interior_points)
loss_bc = boundary_loss(model, boundary_points)

print("Residual shape:", residual.shape)
print("PDE loss:", loss_pde.item())
print("Boundary loss:", loss_bc.item())

