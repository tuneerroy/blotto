import numpy as np
from constants import *

def get_allocations(p, iterations):
    p = p / np.sum(p)
    p_choices = np.random.choice(len(p), (iterations, UNITS), replace=True, p=p)  # (iterations, UNITS)
    p_alloc = np.apply_along_axis(lambda x: np.histogram(x, bins=np.arange(len(p) + 1))[0], axis=1, arr=p_choices)  # (iterations, n)
    return p_alloc

def play(p1_alloc, p2_alloc):
    p1_resources_won, p2_resources_won = np.sum(p1_alloc > p2_alloc, axis=1), np.sum(p2_alloc > p1_alloc, axis=1)  # (iterations,)

    p1_wins = np.sum(p1_resources_won > p2_resources_won)  # scalar
    p2_wins = np.sum(p2_resources_won > p1_resources_won)  # scalar
    ties = np.sum(p1_resources_won == p2_resources_won)  # scalar
    return np.array([p1_wins, p2_wins, ties])

def experts_play(experts, opponent, iterations=1000):
    opponent_alloc = get_allocations(opponent, iterations)
    return np.apply_along_axis(lambda x: play(get_allocations(x, iterations), opponent_alloc), axis=1, arr=experts)

def play_dist(p1, p2, iterations=1000):
    p1_alloc = get_allocations(p1, iterations)
    p2_alloc = get_allocations(p2, iterations)
    return play(p1_alloc, p2_alloc)


def mw(experts: list, opponent, eta=0.1, T=100):
    n = len(experts)
    weights = np.ones(n)
    total_loss = 0
    expert_losses = np.zeros(n)

    for _ in range(T):
        # choose a random expert according to the weights
        random_expert_index = np.random.choice(n, p=weights / weights.sum())

        # play game for each expert against opponent
        results = experts_play(experts, opponent, iterations=1) # (n, 3)
        losses = results[:, 1] + results[:, 2] / 2 # (n,)
        expert_losses += losses
        total_loss += losses[random_expert_index] # update total loss (based on expert chosen)

        # update weights according to losses
        results_expected = experts_play(experts, opponent, iterations=1000) 
        losses_expected = results_expected[:, 1] / (results_expected[:, 0] + results_expected[:, 1])
        weights *= np.exp(1 - eta * losses_expected) 

    return weights, total_loss, expert_losses
