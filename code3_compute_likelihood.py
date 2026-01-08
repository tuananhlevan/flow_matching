import torch
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from torch.distributions.normal import Normal

velocity_model: ModelWrapper = ... # Train the model parameters s.t. model(x_t, t) = ut(xt)

x_1 = torch.randn(batch_size, *data_dim) # Point X1 where we wish to compute log p1(x)

# Define log p0(x)
gaussian_log_density = Normal(torch.zeros(size=data_dim), torch.ones(size=data_dim)).log_prob

solver = ODESolver(velocity_model=velocity_model)
num_steps = 100
x_0, log_p1 = solver.compute_likelihood(
    x_1=x_1,
    method='midpoint',
    step_size=1.0 / num_steps,
    log_p0=gaussian_log_density
)