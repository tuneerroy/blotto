import numpy as np


def get_allocations(p, iterations, UNITS):
    p = p / np.sum(p)
    p_choices = np.random.choice(len(p), (iterations, UNITS), replace=True, p=p)  # (iterations, UNITS)
    p_alloc = np.apply_along_axis(lambda x: np.histogram(x, bins=np.arange(len(p) + 1))[0], axis=1, arr=p_choices)  # (iterations, n)
    return p_alloc

def play(p1_alloc, p2_alloc):
    p1_resources_won, p2_resources_won = np.sum(p1_alloc > p2_alloc, axis=1), np.sum(p2_alloc > p1_alloc, axis=1)  # (iterations,)

    p1_wins = np.sum(p1_resources_won > p2_resources_won)  # scalar
    p2_wins = np.sum(p2_resources_won > p1_resources_won)  # scalar
    ties = np.sum(p1_resources_won == p2_resources_won)  # scalar
    return np.array(p1_wins, p2_wins, ties)

def experts_play(experts, opponent, iterations=1000, UNITS=100):
    opponent_alloc = get_allocations(opponent, iterations, UNITS)
    return np.apply_along_axis(lambda x: play(get_allocations(x, iterations, UNITS), opponent_alloc, iterations, UNITS), axis=1, arr=experts)


def mw(experts: list, opponent, eta=0.1, T=100):
    n = len(experts)
    weights = np.ones(n)
    total_loss = 0
    expert_losses = np.zeros(n)

    for _ in range(T):
        print(f"Current loss: {total_loss}")
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
