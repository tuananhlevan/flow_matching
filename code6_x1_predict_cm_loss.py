import torch
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

path: AffineProbPath = ...
denoiser_model: torch.nn.Module = ...   # Initialize the denoiser
optimizer: torch.optim.Adam(velocity_model.parameters())

for x_0, x_1 in dataloader: # Samples from π0,1 of shape [batch_size, *data_dim]
    t = torch.rand(batch_size)  # Randomize time t ∼ U [0, 1]
    sample = path.sample(t=t, x_0=x_0, x_1=x_1) # Sample the conditional path
    cm_loss = torch.pow(model(sample.x_t, t) - sample.x_1, 2).mean() # CM loss
    optimizer.zero_grad()
    cm_loss.backward()
    optimizer.step()
    
# Convert from denoiser to velocity prediction
class VelocityModel(ModelWrapper):
    def __init__(self, denoiser: torch.nn.Module, path: AffineProbPath):
        super().__init__(model=denoiser)
        self.path = path
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        x_1_prediction = super().forward(x, t, **extras)
        return self.path.target_to_velocity(x_1=x_1_prediction, x_t=x, t=t)
    
# Sample X1
velocity_model = VelocityModel(denoiser=denoiser_model, path=path)
x_0 = torch.randn(batch_size, *data_dim)
solver = ODESolver(velocity_model=velocity_model)
num_steps = 100
x_1 = solver.sample(x_init=x_0, method="midpoint", step_size=1. / num_steps)