import numpy as np


def get_new_expert(experts, weights):
    normalized_weights = weights / np.sum(weights)
    return np.dot(normalized_weights, experts)


def read_in_experts():
    experts = []
    with open("experts.txt") as f:
        for line in f:
            experts.append(np.array([float(x) for x in line.split()]))
    return experts


def dump_experts(experts):
    with open("experts.txt", "w") as f:
        for expert in experts:
            f.write(" ".join([str(x) for x in expert]) + "\n")
