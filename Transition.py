import numpy as np
import random


class Transition:
    def __init__(self, bigger_probability=0.7):
        self.bigger_probability = bigger_probability
        # transition's dimensions are state, action, bigger/smaller probabilities
        self.transition = np.zeros(shape=[6, 2, 2], dtype=np.int32)
        for state in range(0, 3):
            ran = random.random()
            if ran <= 0.5:
                print(state, " ", 0)
                self.transition[state, 0, 0] = state + 3
                self.transition[state, 0, 1] = state
                self.transition[state, 1, 0] = state
                self.transition[state, 1, 1] = state + 3
            else:
                print(state, " ", 1)
                self.transition[state, 1, 0] = state + 3
                self.transition[state, 1, 1] = state
                self.transition[state, 0, 0] = state
                self.transition[state, 0, 1] = state + 3
        for state in range(3, 6):
            ran = random.random()
            if ran <= 0.5:
                print(state, " ", 0)
                self.transition[state, 0, 0] = (state - 1) % 3 + 3
                self.transition[state, 0, 1] = state - 3
                self.transition[state, 1, 0] = state - 3
                self.transition[state, 1, 1] = (state - 1) % 3 + 3
            else:
                print(state, " ", 1)
                self.transition[state, 1, 0] = (state - 1) % 3 + 3
                self.transition[state, 1, 1] = state - 3
                self.transition[state, 0, 0] = state - 3
                self.transition[state, 0, 1] = (state - 1) % 3 + 3

    def step(self, state, action):
        ran = random.random()
        if ran <= self.bigger_probability:
            return self.transition[state, action, 0]
        else:
            return self.transition[state, action, 1]
