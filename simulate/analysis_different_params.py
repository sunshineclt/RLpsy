import pickle

from utils.draw import draw_different_params

randomized = False
SIMULATE_METHOD = "MF_forget"
TRIAL_LENGTH = 144
metrics = "optimal_last"


def draw(forward_planning_value=None):
    if forward_planning_value:
        draw_different_params(all_reduction[forward_planning_value],
                              metrics,
                              "%s under %s condition in simulated algo %s forward %d" % (metrics,
                                                                                         (
                                                                                             "randomized" if randomized else "block"),
                                                                                         SIMULATE_METHOD,
                                                                                         forward_planning_value),
                              smooth=True,
                              trial_length=TRIAL_LENGTH,
                              show=True)
    else:
        draw_different_params(all_reduction,
                              metrics,
                              "%s under %s condition in simulated algo %s" % (metrics,
                                                                              ("randomized" if randomized else "block"),
                                                                              SIMULATE_METHOD),
                              smooth=True,
                              trial_length=TRIAL_LENGTH,
                              show=True)


with open("%s_%s_%s.pkl" % (metrics, SIMULATE_METHOD, ("randomized" if randomized else "block")), "rb") as f:
    all_reduction = pickle.load(f)

if SIMULATE_METHOD.find("MF") > -1:
    draw()
else:
    for i in range(1, 8):
        draw(i)
