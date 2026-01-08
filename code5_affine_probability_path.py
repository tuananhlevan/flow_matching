from flow_matching.path import AffineProbPath, CondOTProbPath
from flow_matching.path.scheduler import (
    CondOTScheduler, PolynomialConvexScheduler, LinearVPScheduler, CosineScheduler
)

# Conditional Optimal Transport schedule with αt = t, σt = 1 − t
path = AffineProbPath(scheduler=CondOTScheduler())
path = CondOTProbPath() # Shorthand for the affine path with the CondOTScheduler above

# Polynomial schedule with αt = tn, σt = 1 − tn
path = AffineProbPath(scheduler=PolynomialConvexScheduler())

# Linear variance preserving schedule with αt = t, σt = √(1 − t2)
path = AffineProbPath(scheduler=LinearVPScheduler())

# Cosine schedule with αt = sin(0.5tπ), σt = cos(0.5tπ)
path = AffineProbPath(scheduler=CosineScheduler())
