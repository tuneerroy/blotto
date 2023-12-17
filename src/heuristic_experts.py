import numpy as np

from constants import *
from mw import *

curr_expert = np.ones(RESOURCES)
experts = [curr_expert]

VARIANCE = 0.025


def get_new_expert(expert, iterations=1000):
    new_expert = expert
    last_ratio, last_expert = 0.5, expert
    momentum = VARIANCE
    for i in range(iterations):
        new_expert = new_expert / np.sum(new_expert)
        new_expert += np.random.normal(0, momentum, RESOURCES)
        new_expert[new_expert < 0] = 0

        new_wins, old_wins, _ = play_dist(new_expert, expert)
        win_ratio = new_wins / (new_wins + old_wins)

        # if win_ratio > 0.75:
        #     return new_expert
        # if i % 100 == 0:
        #     print(i, win_ratio, last_ratio)

        if win_ratio < last_ratio * 0.5:
            print("Reverted", i, win_ratio, last_ratio, momentum)
            new_expert = last_expert
            momentum = VARIANCE
        elif win_ratio > last_ratio * 1.05:
            print("Updated", i, win_ratio, momentum)
            last_ratio, last_expert = win_ratio, new_expert
            new_expert = new_expert * 3
            momentum = VARIANCE
        else:
            momentum += VARIANCE

    return last_expert


new_expert = get_new_expert(curr_expert)
res = play_dist(new_expert, curr_expert)
print(res)
print(res[0] / (res[0] + res[1]))
