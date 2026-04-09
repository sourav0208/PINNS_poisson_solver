import torch 
import torch.nn as nn

mse_loss = nn.MSELoss()

def compute_pde_residual(model, interior_points):
    if not torch.is_grad_enabled():
        raise RuntimeError(
            "Gradient tracking is disabled inside compute_pde_residual(). "
            "Do not call this inside torch.no_grad() or inference_mode()."
        )

    if not interior_points.requires_grad:
        interior_points = interior_points.clone().detach().requires_grad_(True)

    u = model(interior_points)

    if not u.requires_grad:
        raise RuntimeError(
            "Model output does not require grad. "
            "The forward pass likely happened with gradients disabled."
        )

    u = model(interior_points)

    grads = torch.autograd.grad(
        outputs = u,
        inputs = interior_points,
        grad_outputs = torch.ones_like(u),
        create_graph=True,#
        retain_graph=True
    )[0]

    u_x = grads[:,0:1]
    u_y = grads[:,1:2]

    grad_x = torch.autograd.grad(
        outputs=u_x,
        inputs=interior_points,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True
    )[0]

    grad_y = torch.autograd.grad(
        outputs=u_y,
        inputs=interior_points,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True,
        retain_graph=True
    )[0]

    u_xx = grad_x[:,0:1]
    u_yy = grad_y[:,1:2]

    residual = -(u_xx + u_yy) -1.0

    return residual

def pde_loss(model, interior_points):
    residual = compute_pde_residual(model, interior_points)
    target = torch.zeros_like(residual)
    return mse_loss(residual, target)

def boundary_loss(model, boundary_points):
    u_boundary = model(boundary_points)
    target = torch.zeros_like(u_boundary)
    return mse_loss(u_boundary, target)