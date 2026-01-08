import torch
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

class Flow(ModelWrapper):
    def __init__(self, dim=2, h=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + 1, h), torch.nn.ELU(),
            torch.nn.Linear(h, dim)
        )
        
    def forward(self, x, t):
        t = t.view(-1, 1).expand(*x.shape[:-1], -1)
        return self.net(torch.cat((t, x), 1))
        
velocity_model = Flow()

... # Optimize the model parameters s.t. model(x_t, t) = ut(Xt)

x_0 = torch.randn(batch_size, *data_dim)    # Specify the initial condition

solver = ODESolver(velocity_model=velocity_model)
num_steps = 100
x_1 = solver.sample(x_init=x_0, method="mid_point", step_size=1. / num_steps)