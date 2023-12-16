import numpy as np
from numpy.random import choice


class Expert:
    ...


def play_game(experts: list[Expert], player: Expert):
    ...


def get_loss(player: Expert, experts: list[Expert]):
    losses = play_game(experts, player)
    return sum(losses)


v_get_loss = np.vectorize(get_loss)


def mw(experts: list[Expert], eta: float, T: int):
    weights = np.array([1 for _ in experts])
    total_loss = 0
    for _ in range(T):
        # choose a random expert according to the weights
        total_weight = sum(weights)
        random_expert_index = choice(len(experts), p=weights / total_weight)

        # play game for each expert against all experts (including itself)
        losses = v_get_loss(experts, experts)

        # update total loss (based on expert chosen)
        total_loss += losses[random_expert_index]

        # update weights according to losses
        weights *= np.exp(1 - eta * losses)
    return weights, total_loss
