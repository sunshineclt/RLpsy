from psychopy import visual, monitors, core, logging, event

################################################################################
# Monitor setup
################################################################################
# mon = monitors.Monitor("MBP")
# mon.setSizePix([1280, 800])
# mon.setWidth(28.65)
# mon.setDistance(57)

# mon = monitors.Monitor("DellU2414H")
# mon.setSizePix([1920, 1080])
# mon.setWidth(52.70)
# mon.setDistance(57)
################################################################################


################################################################################
# Logging setup
################################################################################
logging.console.setLevel(logging.WARNING)
lastLog = logging.LogFile("lastRun.log", level=logging.INFO, filemode='w')
centralLog = logging.LogFile("AllLog.log", level=logging.WARNING, filemode='a')
################################################################################


################################################################################
# global event key setup
################################################################################
event.globalKeys.clear()


def exit_and_print():
    print("q pressed! exit now...")
    core.quit()


event.globalKeys.add(key='q', func=exit_and_print, name='shutdown')
################################################################################

################################################################################
# Main Program
################################################################################
win = visual.Window(fullscr=True, size=[1280, 800], screen=0, monitor="MBP", units="deg")
# msg = visual.TextStim(win, text="Hello World!")
# msg.setAutoDraw(True)
# win.flip()
# core.wait(1)
#
# msg.setText("23333")
# win.flip()
# core.wait(1)
# msg.setAutoDraw(False)

# create some stimuli
frac = visual.ImageStim(win=win, image="Fractal/julia1.png", size=[5, 5])
fixation = visual.GratingStim(win=win, size=0.2, pos=[0, 0], sf=0)

# draw the stimuli and update the window
while True:  # this creates a never-ending loop
    frac.draw()
    fixation.draw()
    win.flip()

    keys = event.getKeys()
    if len(keys) > 0:
        print(keys)
        break
    event.clearEvents()

win.close()
################################################################################
