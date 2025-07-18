# lightcurve-generator

First: Run MAIN_GS_PASSES.py and select a satellite from active celestrak TLEs (this uses the helper file load_TLEs.py)

The TLEs will only be loaded once every two hours so we dont get banned
Right now I run from the current time in steps of 1 second for 20000 seconds

The code will calculate every pass over the list of ground stations defined further down in the code (above a given elevation) and output a load of parameters to a json.

Second: In Blender, copy and paste the script in to the script window. Alternatively I have uploaded the blender file I'm using.
I've tried to make sure all the relevant settings are scripted but some may remain to be changed. Make sure the output file type is openEXR and that your save directory is correct.

You will need to put in a CAD model and define a material. Those are on the alconcel group meeting onedrive - too big to put here. .obj files (like starlink) have an accompanying .mtl file, that as long as it is in the same directory will apply the material in Blender.
For .stl and others, you will need to define your own material in the shader editor.

Once the script runs without errors and its looking nice, go to render> render animation.

Third: Load the frames using plot_lightcurve.py. This will plot the lightcurve for the contact you selected (in some kind of magnitude - physical meaningfulness TBD)

To do:

Define the attitude of the satellite in the code - right now the rotation is defined in Blender, and therefore starts when the animation does. Should not be too hard to input a few motion parameters to the Blender script via the first script.

Shadow function! Rn the Earth doesnt cast a shadow on the satellite, this is pretty important to implement.

Earth/moonglow?

Plenty of other small things too!
