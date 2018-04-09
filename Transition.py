import numpy as np
import random


class Transition:
    def __init__(self, bigger_probability=0.6):
        self.bigger_probability = bigger_probability
        # transition's dimensions are state, action, bigger/smaller/smaller probabilities
        self.transition = np.zeros(shape=[6, 3, 3], dtype=np.int32)
        for state in range(0, 3):
            dominant = np.array([3, 4, 5])
            np.random.shuffle(dominant)
            for action in range(0, 3):
                self.transition[state, action, 0] = dominant[action]
                print("state: {0}, action: {1}, dominant: {2}".format(state, action, dominant[action]))
                self.transition[state, action, 1] = dominant[(action + 1) % 3]
                self.transition[state, action, 2] = dominant[(action + 2) % 3]
        for state in range(3, 6):
            dominant = np.array([state % 3, (state + 1) % 3 + 3, (state + 2) % 3 + 3])
            np.random.shuffle(dominant)
            for action in range(0, 3):
                self.transition[state, action, 0] = dominant[action]
                print("state: {0}, action: {1}, dominant: {2}".format(state, action, dominant[action]))
                self.transition[state, action, 1] = dominant[(action + 1) % 3]
                self.transition[state, action, 2] = dominant[(action + 2) % 3]

    def step(self, state, action):
        ran = random.random()
        if ran <= self.bigger_probability:
            return self.transition[state, action, 0]
        elif ran <= (1 + self.bigger_probability) / 2:
            return self.transition[state, action, 1]
        else:
            return self.transition[state, action, 2]
