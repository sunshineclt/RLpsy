import pickle

from utils.draw import draw_different_params

randomized = False
SIMULATE_METHOD = "MF"
TRIAL_LENGTH = 144


def draw(forward_planning_value=None):
    if forward_planning_value:
        draw_different_params(all_reduction[forward_planning_value],
                              "optimal",
                              "optimal under %s condition in simulated algo %s forward %d" % (
                                  ("randomized" if randomized else "block"), SIMULATE_METHOD, forward_planning_value),
                              smooth=True,
                              trial_length=TRIAL_LENGTH,
                              show=True)
    else:
        draw_different_params(all_reduction,
                              "optimal",
                              "optimal under %s condition in simulated algo %s" % (
                                  ("randomized" if randomized else "block"), SIMULATE_METHOD),
                              smooth=True,
                              trial_length=TRIAL_LENGTH,
                              show=True)


with open("optimal_last_%s_%s.pkl" % (SIMULATE_METHOD, ("randomized" if randomized else "block")), "rb") as f:
    all_reduction = pickle.load(f)

if SIMULATE_METHOD.find("MF") > -1:
    draw()
else:
    for i in range(1, 8):
        draw(i)
