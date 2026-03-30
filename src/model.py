import torch 
import torch.nn as nn

print("Torch version:", torch.__version__)
print("CUDA runtime in torch:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

class PINN(nn.Module):
    def __init__(self,input_dim=2, hidden_dim=32, num_hidden_layers=3,output_dim=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(num_hidden_layers-1):
            layers.append(nn.Linear(hidden_dim,hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self,x):
        return self.network(x)
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    model = PINN().to(device)
    print(model)

    sample_points = torch.rand(5,2, device=device)
    output = model(sample_points)

    print("Input shape:", sample_points.shape)
    print("Output shape:", output.shape)
    print("Output:", output)
