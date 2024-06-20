from torch import nn

from src.tool.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class MLP(nn.Module):
    def __init__(self, device, input_size: int = 198, hidden_size: int = 256, output_size: int = 198):
        super().__init__()
        self.device = device
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.Sigmoid(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid(),
        )

        # remember to send model to device!
        self.to(self.device)

    def forward(self, x):
        return self.model(x)