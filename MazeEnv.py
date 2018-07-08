import numpy as np
import random
from time import time
from Transition import Transition
from psychopy import data


class MazeEnv:
    def __init__(self, randomized, seed=None):
        self.randomized = randomized
        self.seed = seed
        self.number_of_trials = 1440

        self.trials = 0
        self.task_order = []
        self.transition = Transition()
        self.position = 0
        self.target = 0
        self.task_index = 0
        self.reset_env()

    def reset_env(self):
        if self.seed:
            np.random.seed(self.seed)
            random.seed(self.seed)
        else:
            np.random.seed(int(time()))
            random.seed(int(time()))
        np.random.shuffle([1, 2, 3, 4, 5, 6])

        self.task_order = []
        if self.randomized:
            # randomized
            for i in range(0, self.number_of_trials):
                self.task_order.append({"from": i % 3, "to": (i + 1) % 3})
            np.random.shuffle(self.task_order)
        else:
            # block
            for i in range(0, self.number_of_trials // 4):
                self.task_order.append({"from": 0, "to": 1})
            for i in range(0, self.number_of_trials // 4):
                self.task_order.append({"from": 1, "to": 2})
            for i in range(0, self.number_of_trials // 4):
                self.task_order.append({"from": 2, "to": 0})
            temp_task = []
            for i in range(0, self.number_of_trials // 4):
                temp_task.append({"from": i % 3, "to": (i + 1) % 3})
            np.random.shuffle(temp_task)
            self.task_order.extend(temp_task)

        self.trials = data.TrialHandler(self.task_order, nReps=1, method="sequential", originPath=".")
        self.transition = Transition()

        self.task_index = -1
        self.position = 0
        self.target = 0

    @staticmethod
    def _onehot(position):
        temp = np.zeros(shape=[6])
        temp[position] = 1
        return temp

    def reset(self):
        is_reset = False
        if self.task_index >= self.number_of_trials - 1:
            is_reset = True
            self.reset_env()
        self.task_index += 1
        self.position = self.task_order[self.task_index]["from"]
        self.target = self.task_order[self.task_index]["to"]

        return self._onehot(self.position), is_reset

    def step(self, action, timestep):
        self.position = self.transition.step(self.position, action)
        done = False
        reward = 0
        if self.position == self.target:
            done = True
            reward = max(21 - timestep, 1)
        return self._onehot(self.position), reward, done
