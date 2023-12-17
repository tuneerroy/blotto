from mw import *
from constants import *

N_EXPERTS = 50
T = 100
ETA = np.sqrt(np.log(N_EXPERTS) / T)


experts = np.random.rand(N_EXPERTS, RESOURCES)
_, loss, expert_losses = mw(experts, np.random.rand(RESOURCES), eta=ETA, T=T)

print(f"------------------ETA: {ETA}------------------")

print(f"Loss: {loss}")
print(expert_losses)

print(min(expert_losses))
print(np.log(N_EXPERTS) / ETA + ETA*T)

print(f"----------------------------------------------")
