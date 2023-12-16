import numpy as np


def play(p1, p2, iterations=10000, UNITS=100):
    p1 = p1 / np.sum(p1)
    p2 = p2 / np.sum(p2)
    np.random.shuffle(p1)  # only need to shuffle one
    n = len(p1)
    assert n == len(p2)

    p1_choices = np.random.choice(
        n, (iterations, UNITS), replace=True
    )  # (iterations, UNITS)
    p2_choices = np.random.choice(
        n, (iterations, UNITS), replace=True
    )  # (iterations, UNITS)
    p1_alloc = np.apply_along_axis(
        lambda x: np.histogram(x, bins=np.arange(n + 1))[0], axis=1, arr=p1_choices
    )  # (iterations, n)
    p2_alloc = np.apply_along_axis(
        lambda x: np.histogram(x, bins=np.arange(n + 1))[0], axis=1, arr=p2_choices
    )  # (iterations, n)

    p1_resources_won, p2_resources_won = np.sum(p1_alloc > p2_alloc, axis=1), np.sum(
        p2_alloc > p1_alloc, axis=1
    )  # (iterations,)

    p1_wins = np.sum(p1_resources_won > p2_resources_won)  # scalar
    p2_wins = np.sum(p2_resources_won > p1_resources_won)  # scalar
    ties = np.sum(p1_resources_won == p2_resources_won)  # scalar
    return p1_wins, p2_wins, ties


def get_loss(player, opponent):
    player_wins, opponent_wins, _ = play(player, opponent)
    return opponent_wins / (player_wins + opponent_wins)


v_get_loss = np.vectorize(get_loss)


def mw(experts: list, opponent, eta=0.1, T=1000):
    n = len(experts)
    weights = np.ones(n)
    total_loss = 0

    for _ in range(T):
        # choose a random expert according to the weights
        random_expert_index = np.random.choice(n, p=weights / weights.sum())

        # play game for each expert against all experts (including itself)
        losses = v_get_loss(experts, opponent)

        # update total loss (based on expert chosen)
        total_loss += losses[random_expert_index] > 0.5

        # update weights according to losses
        weights *= np.exp(1 - eta * losses)
    return weights, total_loss
