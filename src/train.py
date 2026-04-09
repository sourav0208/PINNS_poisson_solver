import os
import torch
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import PINN
from data import sample_boundary_points, sample_interior_points
from losses import  pde_loss, boundary_loss, compute_pde_residual

def total_loss_fn(model, interior_points, boundary_points, lambda_bc):
    loss_pde = pde_loss(model, interior_points)
    loss_bc = boundary_loss(model, boundary_points)
    loss_total = loss_pde + lambda_bc*loss_bc

    return loss_pde, loss_bc, loss_total

def train_pinn(
        n_interior=2000,
        n_boundary_per_side=500,
        hidden_dim=64,
        num_hidden_layers=4,
        learning_rate=1e-3,
        adam_epochs =8000,
        lambda_bc=50.0,
        use_lbfgs =True,
        lbfgs_steps=500,
        checkpoint_path=None,
        figure_dir = rf"C:\SOURAV\pinns_poisson_solver\outputs\figures",
        experiment_name = "default"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device:", device)

    if checkpoint_path is None:
         raise ValueError("checkpoint_path must be provided")
    
    checkpoint_dir = os.path.dirname(checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model = PINN(input_dim=2, hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, output_dim=1).to(device)

    total_loss_history = []
    pde_loss_history = []
    bc_loss_history = []

    start_time = time.time()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    progress_bar = tqdm(range(adam_epochs), desc=f"Adam [{experiment_name}]")

    for epoch in progress_bar:
        model.train()
        optimizer.zero_grad()


        interior_points = sample_interior_points(n_interior, device=device)
        boundary_points = sample_boundary_points(n_boundary_per_side, device=device)

        #print("Interior points shape:", interior_points.shape)
        #print("Boundary points shape:", boundary_points.shape)
        #print("interior_points.requires_grad =", interior_points.requires_grad)
        #print("interior_points.device =", interior_points.device)
        #print("interior_points.shape =", interior_points.shape)
        residual_loss = compute_pde_residual(model, interior_points)
        loss_pde, loss_bc, loss_total = total_loss_fn(model, interior_points, boundary_points, lambda_bc)
        
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

    if use_lbfgs:
            model.train()

            interior_points = sample_boundary_points(n_interior, device)
            boundary_points = sample_boundary_points(n_boundary_per_side, device)

            optimizer_lbfgs = torch.optim.LBFGS(
                model.parameters(),
                lr=1.0,
                max_iter=lbfgs_steps,
                max_eval=lbfgs_steps,
                history_size=50,
                line_search_fn="strong_wolfe"
            )
            lbfgs_iteration = [0]

            def closure():
                with torch.enable_grad():
                    optimizer_lbfgs.zero_grad()
                    #print("interior_points.requires_grad =", interior_points.requires_grad)
                    #print("interior_points.device =", interior_points.device)
                    #print("interior_points.shape =", interior_points.shape)

                    residual_loss =compute_pde_residual(model, interior_points)
                    loss_pde, loss_bc, loss_total = total_loss_fn(model, interior_points, boundary_points, lambda_bc)

                    loss_total.backward()

                    total_loss_history.append(loss_total.item())
                    pde_loss_history.append(loss_pde.item())
                    bc_loss_history.append(loss_bc.item())

                    if lbfgs_iteration[0] % 20 == 0:
                        print(
                        f"LBFGS iter {lbfgs_iteration[0]:4d} | "
                        f"total={loss_total.item():.3e}, "
                        f"pde={loss_pde.item():.3e}, "
                        f"bc={loss_bc.item():.3e}"
                    )

                    lbfgs_iteration[0] += 1
                    return loss_total
            
            optimizer_lbfgs.step(closure)

    
    training_time = time.time() - start_time

    print("Saving checkpoint to:", checkpoint_path)
        
    torch.save({
        "model_state_dict": model.state_dict(),
        "hidden_dim": hidden_dim,
        "num_hidden_layers": num_hidden_layers,
        "lambda_bc": lambda_bc,
        "n_interior": n_interior,
        "n_boundary_per_side": n_boundary_per_side,
        "adam_epochs": adam_epochs,
        "use_lbfgs": use_lbfgs,
        "lbfgs_steps": lbfgs_steps
    }, checkpoint_path)

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

    final_total_loss = total_loss_history[-1]
    final_pde_loss = pde_loss_history[-1]
    final_bc_loss = bc_loss_history[-1]

    return {
        "model": model,
        "training_time": training_time,
        "total_loss_history": total_loss_history,
        "pde_loss_history": pde_loss_history,
        "bc_loss_history": bc_loss_history,
        "final_total_loss": final_total_loss,
        "final_pde_loss": final_pde_loss,
        "final_bc_loss": final_bc_loss

    }


if __name__ == "__main__":
    train_pinn()


