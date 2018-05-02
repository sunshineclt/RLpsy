import pickle

from utils.draw import draw_different_params

randomized = True
SIMULATE_METHOD = "MB"
TRIAL_LENGTH = 144


def draw(forward_planning_value):
    draw_different_params(all_reduction[forward_planning_value],
                          "optimal",
                          "optimal under %s condition in simulated algo %s forward %d" % (
                              ("randomized" if randomized else "block"), SIMULATE_METHOD, forward_planning_value),
                          smooth=True,
                          trial_length=TRIAL_LENGTH,
                          show=True)


with open("optimal_MB_block.pkl", "rb") as f:
    all_reduction = pickle.load(f)

for i in range(1, 8):
    draw(i)
