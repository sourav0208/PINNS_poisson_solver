import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import PINN
from visualize import predict_on_grid

def compute_error_metrics(U_pinn,U_fem):
    error= U_pinn - U_fem
    l2_error = np.sqrt(np.mean(error **2))
    real_l2_error = l2_error/np.sqrt(np.mean(U_fem**2))
    max_error = np.max(np.abs(error))

    return{
        "l2_error":l2_error,
        "rel_l2_error":real_l2_error,
        "max_error":max_error,
        "error_field":error
    }

def plot_results(X, Y, U_pinn, U_fem, error, save_dir=rf"outputs\figures"):

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(7, 5))
    contour1 = plt.contourf(X, Y, U_pinn, levels=50)
    plt.colorbar(contour1, label="u_PINN(x,y)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("PINN Solution")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pinn_solution_comparison.png"), dpi=300)
    plt.show()

    plt.figure(figsize=(7, 5))
    contour2 = plt.contourf(X, Y, U_fem, levels=50)
    plt.colorbar(contour2, label="u_FEM(x,y)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("FEM Solution")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fem_solution_comparison.png"), dpi=300)
    plt.show()

    plt.figure(figsize=(7, 5))
    contour3 = plt.contourf(X, Y, error, levels=50)
    plt.colorbar(contour3, label="u_PINN - u_FEM")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Error Field: PINN - FEM")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pinn_fem_error.png"), dpi=300)
    plt.show()

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    checkpoint_path = rf"outputs\checkpoints\pinn_final.pt"   
    checkpoint = torch.load(checkpoint_path, map_location=device) 

    model = PINN(
        hidden_dim=checkpoint["hidden_dim"],
        num_hidden_layers=checkpoint["num_hidden_layers"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    X,Y,U_pinn = predict_on_grid(model, nx=100, ny=100, device=device)

    U_fem = np.load(rf"fem_solution\fem_solution.npy")

    if U_pinn.shape != U_fem.shape:
        raise ValueError(f"Shape mismatch: PINN {U_pinn.shape}, FEM {U_fem.shape}")
    
    metrics = compute_error_metrics(U_pinn, U_fem)
    error = metrics["error_field"]

    print(f"L2 error         : {metrics['l2_error']:.6e}")
    print(f"Relative L2 error: {metrics['rel_l2_error']:.6e}")
    print(f"Max error        : {metrics['max_error']:.6e}")

    plot_results(X, Y, U_pinn, U_fem, error)

if __name__ == "__main__":
    main()
    
