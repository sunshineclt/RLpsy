# -*- coding: utf-8 -*-

from psychopy import gui, core, data, logging, clock, monitors
import numpy as np
from Transition import Transition
import os

################################################################################
# Monitor setup
################################################################################
mon = monitors.Monitor("MBP", width=28.65, distance=57)
mon.setSizePix([1280, 800])
################################################################################


################################################################################
# Logging setup
################################################################################
logging.console.setLevel(logging.DATA)
lastLog = logging.LogFile("lastRun.log", level=logging.INFO, filemode='w')
centralLog = logging.LogFile("AllLog.log", level=logging.WARNING, filemode='a')
################################################################################


################################################################################
# Record participant's information
################################################################################
experiment_name = "MDP"
exp_info = {
    'participant': '',
    'gender': ('male', 'female'),
    'age': '',
    'left-handed': False
}

dlg = gui.DlgFromDict(dictionary=exp_info, title=experiment_name)
if not dlg.OK:
    core.quit()

exp_info["date"] = data.getDateStr()
logging.data(exp_info)
data_path = "data/"
data_fname = exp_info["participant"] + "_" + exp_info["date"]
data_fname = os.path.join(data_path, data_fname)
################################################################################


################################################################################
# global event key setup
################################################################################
from psychopy import event
event.globalKeys.clear()


def exit_and_print():
    print("q pressed! exit now...")
    core.quit()


event.globalKeys.add(key='q', func=exit_and_print, name='shutdown')
################################################################################


################################################################################
# Main Program
################################################################################
from psychopy import visual

win = visual.Window(fullscr=True, size=[1280, 800], screen=0, monitor=mon, units="deg")

# load stimulus images
fractals = []
for i in range(1, 4):
    frac = visual.ImageStim(win, "Fractal/julia" + str(i) + ".png", size=[6, 6])
    fractals.append(frac)

for i in range(2, 5):
    frac = visual.ImageStim(win, "Fractal/mandelbrot" + str(i) + ".png", size=[6, 6])
    fractals.append(frac)

# assign stimulus via pseudo-random across participants
condition_index = int(exp_info["participant"]) % 12
if condition_index == 0:
    index = [0, 1, 2, 3, 4, 5]
elif condition_index == 1:
    index = [0, 1, 2, 3, 4, 5]
else:
    index = [0, 1, 2, 3, 4, 5]
temp = fractals.copy()
for index1, i in enumerate(index):
    fractals[index1] = temp[i]

# prepare for other stimulus
fixation_horizon = visual.Line(win, start=(-.5, 0), end=(.5, 0), lineWidth=3)
fixation_vertical = visual.Line(win, start=(0, -.5), end=(0, .5), lineWidth=3)
# task_indication = visual.TextBox(win, 'In this task, you\'ll be asked to "walk" from a fractal to another. What you need to do is choosing a series of actions to "walk", and finally get to the destination fractal. \n' +
#                                         'Each timestep you can choose one of "f" or "j", and then the pattern will change according to your selection. \n' +
#                                         'It is worth mensioning that action\'s consequence is different under different fractal pattern, and it is stochastic (meaning even choosing the same action under same fractal\'s effect may be different). \n' +
#                                         'Press space to start', pos=(0, 0), size=(5, 5), units="deg", font_size=10)
# task_indication.setSize([5, 5])
task_indication = visual.TextStim(win, "Press space to start")
from_indication = visual.TextStim(win, "From", pos=(0, 7))
to_indication = visual.TextStim(win, "To", pos=(0, -1))
# now_indication = visual.TextStim(win, "Now", pos=(0, 7))
operation_indication = visual.TextStim(win, "Press f or j", pos=(0, 1))
well_done_indication = visual.TextStim(win, "Well Done!", pos=(0, 0))
general_clock = core.Clock()

# indication
task_indication.draw()
win.flip()
event.waitKeys(keyList="space")

# main trial loop
task_order = []
for i in range(0, 200):
    task_order.append({"from": i % 3, "to": (i + 1) % 3})
np.random.shuffle(task_order)
trials = data.TrialHandler(task_order, nReps=1, extraInfo=exp_info, method="sequential", originPath=data_path)
transition = Transition()
steps_record = []
for trial in trials:
    trial_start_state = trial["from"]
    trial_end_state = trial["to"]
    # fixation
    fixation_horizon.draw()
    fixation_vertical.draw()
    win.flip()
    general_clock.reset()
    while general_clock.getTime() < 1:
        fixation_horizon.draw()
        fixation_vertical.draw()
        win.flip()

    # Goal
    fractals[trial_start_state].setPos([0, 3])
    fractals[trial_end_state].setPos([0, -5])
    from_indication.draw()
    to_indication.draw()
    fractals[trial_start_state].draw()
    fractals[trial_end_state].draw()
    win.flip()
    general_clock.reset()
    while general_clock.getTime() < 2:
        from_indication.draw()
        to_indication.draw()
        fractals[trial_start_state].draw()
        fractals[trial_end_state].draw()
        win.flip()
    fractals[trial_start_state].setPos([0, 0])
    fractals[trial_end_state].setPos([0, 0])

    # Start Free Choice
    step = 0
    now = trial_start_state
    trial_record = []
    while now != trial_end_state:
        step += 1
        # now_indication.draw()
        fractals[now].setPos([0, 5])
        fractals[now].draw()
        fractals[now].setPos([0, 0])
        to_indication.draw()
        fractals[trial_end_state].setPos([0, -5])
        fractals[trial_end_state].draw()
        fractals[trial_end_state].setPos([0, 0])
        operation_indication.draw()
        win.flip()

        general_clock.reset()
        rt = 100000
        keys = event.getKeys(keyList=["f", "j"])
        while len(keys) == 0:
            keys = event.getKeys(keyList=["f", "j"])
        rt = general_clock.getTime()
        if keys[0] == 'f':
            action = 0
        else:
            action = 1
        new = transition.step(now, action)
        trial_record.append((now, action, new, rt))
        now = new

    # Feedback
    well_done_indication.setText("Well Done!")
    well_done_indication.setPos([0, 2])
    well_done_indication.draw()
    well_done_indication.setText("It takes you " + str(step) + " steps")
    well_done_indication.setPos([0, 0])
    well_done_indication.draw()
    well_done_indication.setText("Your reward is {}. ".format(max(20 - step, 1)))
    well_done_indication.setPos([0, -2])
    well_done_indication.draw()
    win.flip()
    general_clock.reset()
    while general_clock.getTime() < 3:
        well_done_indication.setText("Well Done!")
        well_done_indication.setPos([0, 2])
        well_done_indication.draw()
        well_done_indication.setText("It takes you " + str(step) + " steps")
        well_done_indication.setPos([0, 0])
        well_done_indication.draw()
        well_done_indication.setText("Your reward is {}. ".format(max(20 - step, 1)))
        well_done_indication.setPos([0, -2])
        well_done_indication.draw()
        win.flip()

    # prepare for next trial
    trials.addData("trial_data", trial_record)
    steps_record.append(step)
    if len(steps_record) >= 5 and np.mean(steps_record[-5:]) <= 10:
        break

trials.saveAsWideText(data_fname + ".csv", delim=",")
win.close()
core.quit()
################################################################################
