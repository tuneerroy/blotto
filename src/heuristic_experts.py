import numpy as np

from constants import *
from mw import *

VARIANCE = 0.01

def generate_expert_heuristic(expert, iterations=200, epoch=5):
    last_ratio, last_expert, last_updated = 0.5, expert, 0
    for i in range(iterations):
        new_expert = last_expert # reset
        if np.random.rand() < 0.3:
            new_expert = new_expert ** 1.1 # encourage more extreme-ness
        momentum = VARIANCE
        for e in range(epoch):
            new_expert = new_expert / np.sum(new_expert)
            new_expert += np.random.normal(0, momentum, RESOURCES)
            new_expert[new_expert < 0] = 0

            new_wins, old_wins, _ = play_dist(new_expert, expert)
            win_ratio = new_wins / (new_wins + old_wins)


            if win_ratio > last_ratio * 1.05:
                print("Updated", i, win_ratio, momentum)
                last_ratio, last_expert, last_updated = win_ratio, new_expert, i
                break
            else:
                momentum += VARIANCE # explore more, the longer we aren't doing better

        if last_updated + 50 <= i and last_ratio > 0.75:
            print("Early stopping")
            # haven't updated in a while and doing pretty well, move on
            break
    
    return last_expert, last_ratio


experts = [np.random.one(RESOURCES)]
for _ in range(100):
    new_expert, ratio = generate_expert_heuristic(experts[-1])
    print(f"Found new expert winning with ratio: {ratio}")
    print(new_expert)
    if ratio > 0.75:
        experts.append(new_expert)