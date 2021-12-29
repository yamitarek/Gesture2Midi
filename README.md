# Gesture2Midi

This Repo has two possible uses: Convert the relative position of your fingers into MIDI-CC commands to control your DAW (Digital Audio Workstation) with gestures OR use them directly inside Pure Data.

## MIDI-CC

1. Run Cam2OSC.py. It should open up a window showing the live camera feed. Try moving your hand to see whether it the hand recognition works. 
  If you ancounter problems, try changing the camera inside the code or the camera width and height.

2. Open OSC2MIDI.pd. You need to have Pure Data installed on your machine for this to work. You shoul see some movement on the faders inside the sketch if you move your hand. You can enable MIDI CC for each finger by using the togglebox above the spigot object.

3. Open your DAW. Make sure to enable the Pure Data Midi In/Out in your DAW and in Pure Data. If you are using Logic Pro X, you can use MIDI Learn to quickly assign gestures/positions to plug-in parameters


## SOUND CREATION 

1. Run Cam2OSC.py. It should open up a window showing the live camera feed. Try moving your hand to see whether it the hand recognition works. 
  If you ancounter problems, try changing the camera inside the code or the camera width and height.

2. Open OSC2Soud.pd. Start with the template and use your imagination or other pd-patches and control them with your fingers.
