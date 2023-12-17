from mw import *

RESOURCES = 10
N_EXPERTS = 20
T = 100
ETA = 0.1

experts = np.random.rand(N_EXPERTS, RESOURCES)
_, loss, expert_losses = mw(experts, np.random.rand(RESOURCES), eta=ETA, T=T)

print(f"Loss: {loss}")
print(expert_losses)

print(min(expert_losses))
print(np.log(N_EXPERTS) / ETA + ETA*T)
