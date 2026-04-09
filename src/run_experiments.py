import os
import csv
import time
import numpy as np
import torch

from train import train_pinn
from model import PINN
from visualize import predict_on_grid

def load_model_from_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = PINN (
        hidden_dim=checkpoint["hidden_dim"],
        num_hidden_layers=checkpoint["num_hidden_layers"]
    ).to(device=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint

def compute_error_metrics(U_pinn,U_fem):
    error= U_pinn - U_fem
    l2_error = np.sqrt(np.mean(error **2))
    real_l2_error = l2_error/np.sqrt(np.mean(U_fem**2))
    max_error = np.max(np.abs(error))

    return l2_error, real_l2_error, max_error

def evaluate_checkpoint_against_fem(checkpoint_path, fem_path, nx=100, ny=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)
    _,_,U_pinn = predict_on_grid(model, nx=nx, ny=ny, device=device)
    U_fem = np.load(fem_path)

    l2_error, rel_l2_error, max_error = compute_error_metrics(U_pinn, U_fem)

    return{
        "hidden_dim": checkpoint["hidden_dim"],
        "num_hidden_layers": checkpoint["num_hidden_layers"],
        "lambda_bc": checkpoint["lambda_bc"],
        "adam_epochs": checkpoint["adam_epochs"],
        "use_lbfgs": checkpoint["use_lbfgs"],
        "lbfgs_steps": checkpoint["lbfgs_steps"],
        "l2_error": l2_error,
        "rel_l2_error": rel_l2_error,
        "max_error": max_error
    }

def main():
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    fem_path = rf"fem_solution\fem_solution.npy"

    experiments = [
        {
            "experiment_name": "hd32_l4_adam_lbfgs",
            "hidden_dim": 32,
            "num_hidden_layers": 4,
            "adam_epochs": 8000,
            "lambda_bc": 50.0,
            "use_lbfgs": True,
            "lbfgs_steps": 500
        },
        {
            "experiment_name": "hd64_l4_adam_lbfgs",
            "hidden_dim": 64,
            "num_hidden_layers": 4,
            "adam_epochs": 8000,
            "lambda_bc": 50.0,
            "use_lbfgs": True,
            "lbfgs_steps": 500
        }
    ]

    results = []

    for exp in experiments:
        #checkpoint_path = os.path.dirname()
        checkpoint_path = rf"outputs\checkpoints\{exp['experiment_name']}.pt"

        print("\n" + "="*60)
        print(f"Running experiment: {exp['experiment_name']}")
        print("="*60)

        train_output = train_pinn(
            n_interior=2000,
            n_boundary_per_side=500,
            hidden_dim=exp["hidden_dim"],
            num_hidden_layers=exp["num_hidden_layers"],
            learning_rate=1e-3,
            adam_epochs=exp["adam_epochs"],
            lambda_bc=exp["lambda_bc"],
            use_lbfgs=exp["use_lbfgs"],
            lbfgs_steps=exp["lbfgs_steps"],
            checkpoint_path=checkpoint_path,
            figure_dir="outputs/figures",
            experiment_name=exp["experiment_name"]
        )

        metrics = evaluate_checkpoint_against_fem(checkpoint_path, fem_path, nx=100, ny=100)
        metrics["experiment_name"] = exp["experiment_name"]
        metrics["training_time_sec"] = train_output["training_time"]
        metrics["final_total_loss"] = train_output["final_total_loss"]
        metrics["final_pde_loss"] = train_output["final_pde_loss"]
        metrics["final_bc_loss"] = train_output["final_bc_loss"]


        results.append(metrics)

        print("Result summary:")
        print(metrics)

    csv_path = rf"outputs\reports\experiment_results.csv"
    fieldnames = [
        "experiment_name",
        "hidden_dim",
        "num_hidden_layers",
        "lambda_bc",
        "adam_epochs",
        "use_lbfgs",
        "lbfgs_steps",
        "l2_error",
        "rel_l2_error",
        "max_error",
        "training_time_sec",
        "final_total_loss",
        "final_pde_loss",
        "final_bc_loss"
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved experiment report to: {csv_path}")


if __name__ == "__main__":
    main()

