import random
import numpy as np


def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


def softmax(x, tau=1):
    """Compute softmax values for each sets of scores in x."""
    temp = x - np.max(x)
    return np.exp(temp * tau) / np.sum(np.exp(temp * tau), axis=0)
