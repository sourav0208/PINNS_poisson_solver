import torch

def sample_interior_points(n_points, device):

    points = torch.rand(n_points, 2, device=device)
    points.requires_grad_(True)
    return points

def sample_boundary_points(n_points_per_side, device):

    y_left = torch.rand(n_points_per_side,1, device=device)
    x_left = torch.zeros_like(y_left)
    left = torch.cat([x_left, y_left], dim =1)

    y_right = torch.rand(n_points_per_side,1, device=device)
    x_right = torch.ones_like(y_right)
    right = torch.cat([x_right, y_right], dim =1)

    x_bottom = torch.rand(n_points_per_side,1, device=device)
    y_bottom = torch.zeros_like(x_bottom)
    bottom = torch.cat([x_bottom, y_bottom], dim =1)

    x_top = torch.rand(n_points_per_side,1, device=device)
    y_top = torch.ones_like(x_top)
    top = torch.cat([x_top, y_top], dim =1)

    boundary_points = torch.cat([left, right, bottom, top], dim =0)

    return boundary_points
