# Multiplicative Weights algorithm

import random

def choose_expert(weights: list[float]):
    total_weight = sum(weights)
    r = random.random()
    for i in range(len(weights)):
        if r < weights[i] / total_weight:
            return i
        r -= weights[i] / total_weight
    return len(weights) - 1