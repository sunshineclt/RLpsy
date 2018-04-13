# -*- coding: utf-8 -*-

import os

import numpy as np
import random
from psychopy import gui, core, data, logging, monitors, sound

from Transition import Transition

################################################################################
# Monitor setup
################################################################################
# mon = monitors.Monitor("MBP", width=28.65, distance=57)
# mon.setSizePix([1280, 800])
mon = monitors.Monitor("iMac", width=47.5967, distance=57)
mon.setSizePix([1920, 1080])
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
logging.info(exp_info)
data_path = "data/"
data_filename = exp_info["participant"] + "_" + exp_info["date"]
data_filename = os.path.join(data_path, data_filename)
################################################################################


################################################################################
# global event key setup
################################################################################
from psychopy import event
event.globalKeys.clear()


def exit_and_print():
    print("q pressed! exit now...")
    trials.saveAsWideText(data_filename + "_quit" + ".csv", delim="#")
    core.quit()


event.globalKeys.add(key='q', func=exit_and_print, name='shutdown')
################################################################################


################################################################################
# Main Program
################################################################################
from psychopy import visual

win = visual.Window(fullscr=True, size=[1920, 1080], screen=0, monitor=mon, units="deg")
np.random.seed(int(exp_info["participant"]))
random.seed(int(exp_info["participant"]))


# load stimulus images
fractals = []
for i in range(1, 7):
    frac = visual.ImageStim(win, "Fractal/" + str(i) + ".jpg", size=[6, 6])
    fractals.append(frac)
# assign stimulus via random across participants
np.random.shuffle(fractals)

# prepare for other stimulus
fixation_horizon = visual.Line(win, start=(-.5, 0), end=(.5, 0), lineWidth=3)
fixation_vertical = visual.Line(win, start=(0, -.5), end=(0, .5), lineWidth=3)
# task_indication = visual.TextBox(win, 'In this task, you\'ll be asked to "walk" from a fractal to another. What you need to do is choosing a series of actions to "walk", and finally get to the destination fractal. \n' +
#                                         'Each timestep you can choose one of "f" or "j", and then the pattern will change according to your selection. \n' +
#                                         'It is worth mensioning that action\'s consequence is different under different fractal pattern, and it is stochastic (meaning even choosing the same action under same fractal\'s effect may be different). \n' +
#                                         'Press space to start', pos=(0, 0), size=(5, 5), units="deg", font_size=10)
# task_indication.setSize([5, 5])
task_label = visual.TextStim(win, "Press space to start")
from_label = visual.TextStim(win, "From", pos=(0, 5))
to_label = visual.TextStim(win, "To", pos=(11, 5))
operation_label = visual.TextStim(win, "Press f or j", pos=(0, 1))
operation_label = [visual.Line(win, start=(0, 8), end=(1, 6.268), lineWidth=5),
                   visual.Line(win, start=(0, 8), end=(-1, 6.268), lineWidth=5),
                   visual.Line(win, start=(-6.928, -5), end=(-4.996, -4.482), lineWidth=5),
                   visual.Line(win, start=(-6.928, -5), end=(-6.411, -3.068), lineWidth=5),
                   visual.Line(win, start=(6.928, -5), end=(4.996, -4.482), lineWidth=5),
                   visual.Line(win, start=(6.928, -5), end=(6.411, -3.068), lineWidth=5)]
reward_label = visual.TextStim(win, "Reward: ", pos=(0, -7))
well_done_label = visual.TextStim(win, "Well Done!", pos=(0, -6))
general_clock = core.Clock()
success_sound = sound.Sound("success.wav")

# indication
task_label.draw()
win.flip()
event.waitKeys(keyList="space")

# generate randomized trial sequence
task_order = []
for i in range(0, 150):
    task_order.append({"from": i % 3, "to": (i + 1) % 3})
np.random.shuffle(task_order)
trials = data.TrialHandler(task_order, nReps=1, extraInfo=exp_info, method="sequential", originPath=data_path)

# main trial loop
transition = Transition()
timesteps_record = []
for trial in trials:
    trial_start_state = trial["from"]
    trial_end_state = trial["to"]
    # fixation
    fixation_horizon.draw()
    fixation_vertical.draw()
    # win.getMovieFrame(buffer='back')
    # win.saveMovieFrames('fixation.png')
    win.flip()
    general_clock.reset()
    while general_clock.getTime() < 1:
        fixation_horizon.draw()
        fixation_vertical.draw()
        win.flip()
    win.flip()

    # goal
    fractals[trial_start_state].setPos([0, 0])
    fractals[trial_end_state].setPos([11, 0])
    # from_label.draw()
    # to_label.draw()
    # fractals[trial_start_state].draw()
    # fractals[trial_end_state].draw()
    # win.getMovieFrame(buffer='back')
    # win.saveMovieFrames('indication.png')
    general_clock.reset()
    while general_clock.getTime() < 1.5:
        from_label.draw()
        to_label.draw()
        fractals[trial_start_state].draw()
        fractals[trial_end_state].draw()
        win.flip()

    # Start Free Choice
    step = 0
    now_state = trial_start_state
    trial_record = []
    while now_state != trial_end_state:
        # print("now: ", now_state)
        step += 1

        # show state
        fractals[now_state].setPos([0, 0])
        fractals[now_state].draw()
        fractals[trial_end_state].setPos([11, 0])
        fractals[trial_end_state].draw()
        to_label.draw()
        [i.draw() for i in operation_label]
        reward_label.setText("Reward: " + str(max(21 - step, 1)))
        reward_label.draw()
        # if step == 1:
        #     win.getMovieFrame(buffer='back')
        #     win.saveMovieFrames('action.png')
        win.flip()

        # listen to response
        general_clock.reset()
        rt = 100000
        event.clearEvents()
        keys = event.getKeys(keyList=["b", "n", "h"], timeStamped=general_clock)
        while len(keys) == 0:
            keys = event.getKeys(keyList=["b", "n", "h"], timeStamped=general_clock)
        rt = keys[0][1]
        if keys[0][0] == 'h':
            action = 0
        elif keys[0][0] == 'b':
            action = 1
        else:
            action = 2

        # emphasize action selection
        operation_label[action * 2].setOpacity(0)
        operation_label[action * 2 + 1].setOpacity(0)
        fractals[now_state].setPos([0, 0])
        fractals[now_state].draw()
        fractals[trial_end_state].setPos([11, 0])
        fractals[trial_end_state].draw()
        to_label.draw()
        [i.draw() for i in operation_label]
        reward_label.draw()
        win.flip()
        general_clock.reset()
        while general_clock.getTime() < 0.1:
            pass
        operation_label[action * 2].setOpacity(1)
        operation_label[action * 2 + 1].setOpacity(1)
        fractals[now_state].setPos([0, 0])
        fractals[now_state].draw()
        fractals[trial_end_state].setPos([11, 0])
        fractals[trial_end_state].draw()
        to_label.draw()
        [i.draw() for i in operation_label]
        reward_label.draw()
        win.flip()

        # make transition
        new_state = transition.step(now_state, action)
        trial_record.append([now_state, action, new_state, rt])
        for i in range(0, 21):
            fractals[new_state].setPos([0, 0])
            fractals[new_state].setOpacity(i / 20)
            fractals[new_state].draw()
            fractals[now_state].setPos([0, 0])
            fractals[now_state].setOpacity(1 - i / 20)
            fractals[now_state].draw()
            fractals[trial_end_state].setPos([11, 0])
            fractals[trial_end_state].setOpacity(1)
            fractals[trial_end_state].draw()
            to_label.draw()
            [i.draw() for i in operation_label]
            reward_label.draw()
            win.flip()
        fractals[now_state].setOpacity(1)
        now_state = new_state

    # Feedback
    success_sound.play()
    # fractals[now_state].setPos([0, 0])
    # fractals[now_state].draw()
    # [i.draw() for i in operation_label]
    # reward_label.draw()
    # well_done_label.setText("Well Done!")
    # well_done_label.setPos([0, -6])
    # well_done_label.draw()
    # well_done_label.setText(str(step) + " steps")
    # well_done_label.setPos([0, -8])
    # well_done_label.draw()
    # win.getMovieFrame(buffer='back')
    # win.saveMovieFrames('feedback.png')
    general_clock.reset()
    while general_clock.getTime() < 3:
        fractals[now_state].setPos([0, 0])
        fractals[now_state].draw()
        [i.draw() for i in operation_label]
        reward_label.draw()
        well_done_label.setText("Well Done!")
        well_done_label.setPos([0, -6])
        well_done_label.draw()
        well_done_label.setText(str(step) + " steps")
        well_done_label.setPos([0, -8])
        well_done_label.draw()
        win.flip()
    win.flip()

    # data storing
    trials.addData("trial_data", trial_record)
    timesteps_record.append(step)
    # if len(timesteps_record) >= 10 and np.mean(timesteps_record[-5:]) <= 3:
    #     break
    print("trial length: ", step)

trials.saveAsWideText(data_filename + ".csv", delim="#")
win.close()
core.quit()
################################################################################
