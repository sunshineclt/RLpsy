import numpy as np
import random
from Transition import Transition


def optimal_probability(seed, trials, is_simulate=False, is_randomized=False):
    np.random.seed(seed)
    random.seed(seed)
    np.random.shuffle([1, 2, 3, 4, 5, 6])
    if (seed % 2 == 0 and not is_simulate) or (is_randomized and is_simulate):
        np.random.shuffle(np.arange(0, 144))
    else:
        np.random.shuffle(np.arange(0, 36))
    transition = Transition()
    result_state = np.zeros(shape=[6, 3])
    for state in range(0, 6):
        for action in range(0, 3):
            result_state[state, action] = transition.transition[state, action, 0]
    action_should_choose = np.zeros(shape=[6, 6])
    action_should_choose -= 1
    for state in range(0, 6):
        for next_state in range(0, 6):
            if state == next_state:
                continue
            for action in range(0, 3):
                if result_state[state, action] == next_state:
                    action_should_choose[state, next_state] = action
    optimal = np.zeros(shape=[6, 3])
    for destination in range(0, 3):
        for state in range(0, 6):
            if state == destination:
                continue
            if state == destination + 3:
                optimal[state, destination] = action_should_choose[state, destination]
            else:
                optimal[state, destination] = action_should_choose[state, destination + 3]

    optimal_ps = []
    optimal_ps_grouped = {"inner": [], "outer": []}
    for trial in trials:
        dest = trial[-1][2]
        optimal_count = 0
        optimal_inner_count = 0
        inner_count = 0
        for step in trial:
            if step[1] == optimal[step[0], dest]:
                optimal_count += 1
                if step[0] > 2:
                    optimal_inner_count += 1
            if step[0] > 2:
                inner_count += 1

        optimal_p = optimal_count / len(trial)
        optimal_ps.append(optimal_p)
        optimal_ps_grouped["inner"].append(optimal_inner_count / inner_count)
        optimal_ps_grouped["outer"].append((optimal_count - optimal_inner_count) /
                                           (len(trial) - inner_count))

    return optimal_ps, optimal_ps_grouped


def calculate_optimal(transition):
    result_state = np.zeros(shape=[6, 3])
    for state in range(0, 6):
        for action in range(0, 3):
            result_state[state, action] = transition.transition[state, action, 0]
    action_should_choose = np.zeros(shape=[6, 6])
    action_should_choose -= 1
    for state in range(0, 6):
        for next_state in range(0, 6):
            if state == next_state:
                continue
            for action in range(0, 3):
                if result_state[state, action] == next_state:
                    action_should_choose[state, next_state] = action
    optimal = np.zeros(shape=[6, 3], dtype=np.int32)
    optimal -= 1
    for destination in range(0, 3):
        for state in range(0, 6):
            if state == destination:
                continue
            if state == destination + 3:
                optimal[state, destination] = action_should_choose[state, destination]
            else:
                optimal[state, destination] = action_should_choose[state, destination + 3]
    return optimal


if __name__ == "__main__":
    optimal_probability(14, None)
