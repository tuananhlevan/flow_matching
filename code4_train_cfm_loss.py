import torch
from flow_matching.path import ProbPath
from flow_matching.path.path_sample import PathSample

path: ProbPath = ... # The flow_matching library implements the most common probability paths
velocity_model: torch.nn.Module = ... # Initialize the velocity model
optimizer = torch.optim.Adam(velocity_model.parameters())

for x_0, x_1 in dataloader: # Samples from π0,1 of shape [batch_size, *data_dim]
    t = torch.rand(batch_size) # Randomize time t ∼ U [0, 1]
    sample: PathSample = path.sample(t=t, x_0=x_0, x_1=x_1)
    x_t = sample.x_t
    dx_t = sample.dx_t # dX_t is ψ ̇t(X0|X1).
    # If D is the Euclidean distance, the CFM objective corresponds to the mean-squared error
    cfm_loss = torch.pow(velocity_model(x_t, t) - dx_t, 2).mean() # Monte Carlo estimate
    optimizer.zero_grad()
    cfm_loss.backward()
    optimizer.step()