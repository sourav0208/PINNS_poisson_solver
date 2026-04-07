import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def generate_evaluation_grid(nx=100, ny=100, device="cpu"):

    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)

    X,Y = np.meshgrid(x,y)

    points = np.column_stack([X.ravel(), Y.ravel()])
    points_tensor = torch.tensor(points, dtype=torch.float32, device=device)

    return X, Y, points_tensor

def predict_on_grid(model, nx=100, ny=100, device="cpu"):

    model.eval()
    X, Y, points_tensor = generate_evaluation_grid(nx=nx, ny=ny, device=device)

    with torch.no_grad():
        u_pred =  model(points_tensor)

    U = u_pred.detach().cpu().numpy().reshape(ny,nx)

    return X,Y,U

def plot_solution(X,Y,U,save_path=None,show=True):

    plt.figure(figsize=(7,5))
    contour = plt.contourf(X,Y,U, levels=50)
    plt.colorbar(contour, label="u(x,y)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("PINN Predicted Solution")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=600)

    if show:
        plt.show()
    else:
        plt.close()

def plot_solution_3d(X,Y,U, save_path=None, show=True):

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(X,Y,U)
    fig.colorbar(surface, ax=ax, shrink=0.75, label="u(x,y)")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    ax.set_title("PINN Predicted Solution Surface")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=600)

    if show:
        plt.show()
    else:
        plt.close()


def save_prediction_array(U, save_path):

    np.save(save_path,U)