import numpy as np
from psychopy import data
from Transition import Transition
import random

task_order = []
number_of_trials = 144

# randomized
for i in range(0, number_of_trials):
    task_order.append({"from": i % 3, "to": (i + 1) % 3})
np.random.shuffle(task_order)

# block
# for i in range(0, 36):
#     task_order.append({"from": 0, "to": 1})
# for i in range(0, 36):
#     task_order.append({"from": 1, "to": 2})
# for i in range(0, 36):
#     task_order.append({"from": 2, "to": 0})
# temp_task = []
# for i in range(0, 36):
#     temp_task.append({"from": i % 3, "to": (i + 1) % 3})
# np.random.shuffle(temp_task)
# task_order.extend(temp_task)

trials = data.TrialHandler(task_order, nReps=1, method="sequential", originPath=".")

transition = Transition()
timesteps_record = []
episode = 0
total_reward = 0
for trial in trials:
    episode += 1

    trial_start_state = trial["from"]
    trial_end_state = trial["to"]

    # Start Free Choice
    step = 0
    now_state = trial_start_state
    trial_record = []
    while now_state != trial_end_state:
        # print("now: ", now_state)
        step += 1

        action = random.randint(0, 2)

        new_state = transition.step(now_state, action)
        trial_record.append([now_state, action, new_state])
        now_state = new_state

    # raw_data storing
    total_reward += max(21 - step, 1)
    trials.addData("trial_data", trial_record)
    timesteps_record.append(step)
    # if len(timesteps_record) >= 10 and np.mean(timesteps_record[-5:]) <= 3:
    #     break
    print("trial length: ", step)

trials.saveAsWideText("data/random/4_simulate.csv", delim="#")
print("Total Reward: ", total_reward)
