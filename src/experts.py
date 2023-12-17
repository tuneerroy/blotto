import numpy as np

from mw import mw


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


def create_more_experts(num_new: int):
    experts = read_in_experts()
    for _ in range(num_new):
        weights, _ = mw(experts, experts)
        new_expert = get_new_expert(experts, weights)
        experts.append(new_expert)
    dump_experts(experts)
