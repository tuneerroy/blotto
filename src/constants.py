import numpy as np

UNITS = 100
RESOURCES = 10
N_EXPERTS = 50
T = 100
ETA = np.sqrt(np.log(N_EXPERTS) / T)
