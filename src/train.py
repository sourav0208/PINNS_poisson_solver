import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import PINN
from data import sample_boundary_points, sample_interior_points
from losses import compute_pde_residual, pde_loss, boundary_loss


def train_pinn(
        n_interior=2000,
        n_boundary_per_side=500,
        hidden_dim=32,
        num_hidden_layers=3,
        lerning_rate=1e-3,
        epochs=5000,
        lambda_bc=10.0,
        chechpoint_dir=rf"C:\SOURAV\pinns_poisson_solver\outputs\checkpoints",
        figure_dir = rf"C:\SOURAV\pinns_poisson_solver\outputs\figures"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device:", device)

    os.makedirs(chechpoint_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    

    model = PINN(input_dim=2, hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, output_dim=1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lerning_rate)

    total_loss_history = []
    pde_loss_history = []
    bc_loss_history = []

    progress_bar = tqdm(range(epochs), desc="Training PINN")

    for epoch in progress_bar:
        model.train()
        optimizer.zero_grad()


        interior_points = sample_interior_points(n_interior, device=device)
        boundary_points = sample_boundary_points(n_boundary_per_side, device=device)

        #print("Interior points shape:", interior_points.shape)
        #print("Boundary points shape:", boundary_points.shape)

        residual = compute_pde_residual(model, interior_points)
        loss_pde = pde_loss(model, interior_points)
        loss_bc = boundary_loss(model, boundary_points)
        loss_total = loss_pde + lambda_bc*loss_bc

        loss_total.backward()
        optimizer.step()

        total_loss_history.append(loss_total.item())
        pde_loss_history.append(loss_pde.item())
        bc_loss_history.append(loss_bc.item())

        progress_bar.set_postfix({
            "total": f"{loss_total.item():.3e}",
            "pde":f"{loss_pde.item():.3e}",
            "bc":f"{loss_bc.item():.3e}"
        })

        if (epoch+1) % 1000 == 0:
            checkpoint_path = os.path.join(chechpoint_dir, f"pinn_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)

    final_model_path = os.path.join(chechpoint_dir, "pinn_final.pt")
    torch.save(model.state_dict(), final_model_path)

    plt.figure(figsize=(8,5))
    plt.semilogy(total_loss_history, label="Total Loss")
    plt.semilogy(pde_loss_history, label="PDE Loss")
    plt.semilogy(bc_loss_history, label="Boundary Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PINN Training loss history")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    figure_path = os.path.join(figure_dir, "loss_history.png")
    plt.savefig(figure_path, dpi=300)
    plt.show()

    return model, total_loss_history, pde_loss_history, bc_loss_history


if __name__ == "__main__":
    train_pinn()


