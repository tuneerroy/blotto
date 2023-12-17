import os

import numpy as np

from constants import *
from mw import *
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


VARIANCE = 0.025


def create_more_experts(num_new: int, starting_experts=10):
    experts = np.array(read_in_experts())
    if len(experts) < starting_experts:
        experts = np.random.rand(starting_experts, RESOURCES)
        num_new -= len(experts)

    best_experts = np.array([])
    min_loss = float("inf")

    for i in range(num_new):
        print("Creating new expert: ", i + 1)
        if len(best_experts) > 0 and np.random.rand() < 0.3:
            opponent = best_experts[np.random.choice(range(len(best_experts)))]
        else:
            opponent = np.random.rand(RESOURCES)

        # randomly choose a random amount of experts
        num_experts = np.random.choice(range(1, len(experts) - 1))
        target_experts = experts[np.random.choice(range(len(experts)), num_experts)]

        weights, loss, _ = mw(target_experts, opponent, eta=ETA, T=T)
        new_expert = get_new_expert(target_experts, weights)

        if loss < min_loss:
            min_loss = loss
            best_experts = (
                np.vstack((best_experts, new_expert))
                if len(best_experts) > 0
                else np.array([new_expert])
            )

        print("Loss: ", loss, " Min loss: ", min_loss)

        # randomly perturb the new expert
        new_expert += np.random.normal(0, VARIANCE, RESOURCES)
        new_expert[new_expert < 0] = 0
        new_expert = new_expert / np.sum(new_expert)

        experts = np.vstack((experts, new_expert))
        print("New expert created...")
        print("-" * 50)

        if i % 5 == 0:
            dump_experts(experts)

    dump_experts(experts)


if __name__ == "__main__":
    create_more_experts(50)
