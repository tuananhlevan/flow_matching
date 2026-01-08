import torch
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import ScheduleTransformedModel, CondOTScheduler, VPScheduler
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

training_scheduler = VPScheduler()  # Variance preserving schedule
path = AffineProbPath(scheduler=training_scheduler)
velocity_model: ModelWrapper = ...  # Train a velocity model with the variance preserving schedule

# Change the scheduler from variance preserving to conditional OT schedule
sampling_scheduler = CondOTScheduler()
transformed_model = ScheduleTransformedModel(
    velocity_model=velocity_model,
    original_scheduler=training_scheduler,
    new_scheduler=sampling_scheduler,
)

# Sample the transformed model with the conditional OT schedule
solver = ODESolver(velocity_model=transformed_model)
x_0 = torch.randn(batch_size, *data_dim)    # Specify the initial condition
solver = ODESolver(velocity_model=velocity_model)
num_steps = 100
x_1 = solver.sample(x_init=x_0, method="midpoint", step_size=1. / num_steps)