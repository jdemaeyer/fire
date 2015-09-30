import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from fire.manager import ProcessingManager


# CONFIG

justpreproc = False

# filenames = ["img1.jpg", "img2.jpg", "img3.jpg"]
filename = "example_img.jpg"
configfile = "extra/example_config.yml"


# SCRIPT

pm = ProcessingManager()

if justpreproc:
    pm.loadconfig(configfile)
    img = pm.loadimage(filename)
    img = pm.preprocess(img, pm.preprocsteps)
    import numpy as np
    np.save("example_img_preproc.npy", img)
else:
    try:
        pm.process(configfile, filenames, overwrite=True)
    except NameError:
        pm.process(configfile, [filename], overwrite=True)
