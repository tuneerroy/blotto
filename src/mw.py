# Multiplicative Weights algorithm

from abc import ABC, abstractmethod
import random

class Expert(ABC):  
    @abstractmethod
    def get_answer(self) -> int:
        raise NotImplementedError


def choose_expert(weights: list[float]):
    total_weight = sum(weights)
    r = random.random()
    for i in range(len(weights)):
        if r < weights[i] / total_weight:
            return i
        r -= weights[i] / total_weight
    return len(weights) - 1

def get_loss(results: list[int]):
    ...

def mw(experts: list[Expert], eta: float, T: int, run_game: callable):
    weights = [1 for _ in experts]
    total_loss = 0
    for _ in range(T):
        expert_index = choose_expert(weights)
        losses = [0 for _ in experts]
        for i, expert in enumerate(experts):
            ans = expert.get_answer()
            results = run_game(ans)
            losses[i] = get_loss(results)

        for i in range(len(weights)):
            weights[i] *= 1 - eta * losses[i]
        
        total_loss += losses[expert_index]
    return weights, total_loss