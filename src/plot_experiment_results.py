import os
import pandas as pd
import matplotlib.pyplot as plt

csv_path = rf"outputs\reports\experiment_results.csv"
figure_dir = rf"outputs\figures"
os.makedirs(figure_dir, exist_ok=True)

df = pd.read_csv(csv_path)

plt.figure(figsize=(7, 5))
plt.bar(df["experiment_name"], df["rel_l2_error"])
plt.ylabel("Relative L2 Error")
plt.title("PINN Width Comparison")
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, "comparison_rel_l2_error.png"), dpi=300)
plt.show()

plt.figure(figsize=(7, 5))
plt.bar(df["experiment_name"], df["training_time_sec"])
plt.ylabel("Training Time [s]")
plt.title("PINN Training Time Comparison")
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, "comparison_training_time.png"), dpi=300)
plt.show()

plt.figure(figsize=(7, 5))
plt.bar(df["experiment_name"], df["final_total_loss"])
plt.ylabel("Final Total Loss")
plt.title("PINN Final Loss Comparison")
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, "comparison_final_loss.png"), dpi=300)
plt.show()