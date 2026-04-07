import os
import torch

from model import PINN
from visualize import predict_on_grid, plot_solution, plot_solution_3d, save_prediction_array

def main():
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    checkpoint_path = rf"outputs\checkpoints\pinn_final.pt"
    figure_dir = rf"outputs\figures"
    os.makedirs(figure_dir, exist_ok=True)

    model = PINN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    X,Y,U = predict_on_grid(model, nx=100, ny=100, device=device)

    plot_solution(
        X,Y,U,
        save_path=os.path.join(figure_dir, "pinn_solution_contour.png"),
        show=True
    )

    plot_solution_3d(
        X,Y,U,
        save_path=os.path.join(figure_dir, "pinn_solution_surface.png"),
        show=True

    )

    save_prediction_array(U,os.path.join(figure_dir, "pinn_solution.npy") )


if __name__ == "__main__":
    main()