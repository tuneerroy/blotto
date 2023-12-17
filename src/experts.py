from mw import *
from constants import *
import os
import numpy as np

from mw import mw


def get_new_expert(experts, weights):
    normalized_weights = weights / np.sum(weights)
    return np.dot(normalized_weights, experts)


def read_in_experts():
    if not os.path.exists("experts.txt"):
        return []

    experts = []
    with open("experts.txt") as f:
        for line in f:
            experts.append(np.array([float(x) for x in line.split()]))
    return experts


def dump_experts(experts):
    with open("experts.txt", "w") as f:
        for expert in experts:
            f.write(" ".join([str(x) for x in expert]) + "\n")


def create_more_experts(num_new: int, starting_experts=10):
    
    experts = np.array(read_in_experts())
    if len(experts) < starting_experts:
        experts = np.random.rand(starting_experts, RESOURCES)
        starting_experts -= len(experts)

    min_loss = float("inf")
    for i in range(num_new):
        print("Creating new expert: ", i + 1)
        opponent = np.random.rand(RESOURCES)
        weights, loss, _ = mw(experts, opponent, eta=ETA, T=T)
        min_loss = min(min_loss, loss)
        print("Loss: ", loss, " Min loss: ", min_loss)
        new_expert = get_new_expert(experts, weights)
        experts = np.vstack((experts, new_expert))
        print("New expert created...")
        print("-" * 50)

    dump_experts(experts)

if __name__ == "__main__":
    create_more_experts(50)