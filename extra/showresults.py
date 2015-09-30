import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import OIP.imageloaders
import OIP.preprocessors
import OIP.exporters
import OIP.detectioninfos
from OIP.debug import randlabels


# CONFIG

#imgfolders = [ "/run/media/jakob/TOSHIBA EXT/Data/Small/20150210_04-08/04/",
#        "/run/media/jakob/TOSHIBA EXT/Data/Small/20150210_04-08/05/",
#        "/run/media/jakob/TOSHIBA EXT/Data/Small/20150210_04-08/07/",
#        "/run/media/jakob/TOSHIBA EXT/Data/Small/20150210_04-08/08/" ]
#crop = { 'y': 66, 'x': 1170, 'dy': 2712, 'dx': 2712 }
#crop = { 'y': 66+1368-1240, 'x': 1170+1350-1240, 'dy': 2*1240, 'dx': 2*1240 }
#crop = { 'y': 66+1368-1240+1370, 'x': 1170+1350-1240+1200, 'dy': 500, 'dx': 700 }
imgfolders = [ "testdata/" ]
crop = { 'y': 252, 'x': 942, 'dx': 2370, 'dy': 2370 }
#crop = { 'y': 552, 'x': 1242, 'dy': 500, 'dx': 500 }
#crop = { 'y': 1252, 'x': 2042, 'dy': 500, 'dx': 500 }
imgfolders = [
    "/run/media/jakob/TOSHIBA EXT/Data/Big/20150218/Full/100NCD90/",
    "/run/media/jakob/TOSHIBA EXT/Data/Big/20150220/Full/100NCD90/",
    "/run/media/jakob/TOSHIBA EXT/Data/Big/20150225/Full/100NCD90/",
    "/run/media/jakob/TOSHIBA EXT/Data/Big/20150220/Full/100NCD90/",
    ]
#crop = { 'y': 752, 'x': 1442, 'dy': 1000, 'dx': 1000 }
#imgfolders = [ "testdata/" ]
#crop = { 'y': 1052, 'x': 2000, 'dy': 300, 'dx': 300 }



# PROCESS CONFIG

imagefiles = []
for imgfolder in imgfolders:
    imagefiles.extend(
        sorted([ os.path.join(imgfolder, f[:-4])
          for f in os.listdir(os.path.join(imgfolder, "processed"))
          if os.path.isfile(os.path.join(imgfolder, "processed", f))
          and f.endswith(".txt") ])
        )


# PREPARE WINDOWS

fig = plt.figure()
fig.subplots_adjust(.01, .01, .99, .99)
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)


# HELPER FUNCTIONS

def loadimg(imgfile):
    global crop, fig, img, im
    print "Loading {}".format(imgfile)
    img = OIP.imageloaders.MahotasImageLoader().load(imgfile)
    img = OIP.preprocessors.CropPreprocessor()(img, **crop)
    im = plt.imshow(img, interpolation = 'none',
                    cmap = plt.cm.Greys_r)
    plt.xlim(xmin=0, xmax = img.shape[1])
    plt.ylim(ymax=0, ymin = img.shape[0])
    fig.canvas.draw()
    #plt.colorbar()

def removedetections():
    global detection_artists
    for da in detection_artists:
        da.remove()
    detection_artists = []

def drawdetections(imgfile):
    global detection_artists, ax
    removedetections()
    # Get list of exported files
    procfilebase = OIP.exporters.outfilebase(imgfile)
    procfolder = os.path.dirname(procfilebase)
    procfiles = [ os.path.join(procfolder, f )
                  for f in os.listdir(procfolder)
                  if os.path.isfile(os.path.join(procfolder, f))
                  and os.path.join(procfolder, f).startswith(procfilebase)
                ]
    allnofais = None
    for procfile in procfiles:
        print "Processing {}".format(procfile)
        with open(procfile, 'r') as f:
            if procfile.endswith('.txt'):
                for line in f.read().splitlines():
                    splitline = line.split("\t")
                    if splitline[0] == 'd':
                        # Disk!
                        x, y, r = ( float(bit) for bit in splitline[2:] )
                        da = plt.Circle((y, x), r, color = 'r',
                                        fill = False, lw = 1.)
                        ax.add_artist(da)
                        detection_artists.append(da)
            elif procfile.endswith('.npz'):
                try:
                    nofai = OIP.detectioninfos.NonOverlappingFreeAreasInfo.load(
                                                                    procfile)
                except KeyError:
                    print "Error: Unable to extract NOFAI"
                    continue
                if allnofais is None:
                    allnofais = nofai.labeledareas.astype(np.uint16)
                else:
                    nofai.labeledareas[np.where(nofai.labeledareas != 0)] += allnofais.max()
                    allnofais += nofai.labeledareas
    if allnofais is not None:
        # Remove bigger than 200,000 pixels
        #sizes = np.bincount(allnofais.flat)
        nrofareas = allnofais.max()
        #for label in np.argwhere(sizes > 150000).flatten():
        #    if label == 0:  # ignore background
        #        continue
        #    print "Removing label {}, size {}".format(label, sizes[label])
        #    allnofais[allnofais == label] = 0
        #    nrofareas -= 1
        randlabels(allnofais)
        plt.imshow(allnofais, alpha = 0.2) 
    print "Drawing {} structures ({} circles and {} areas)".format(len(detection_artists)+nrofareas, len(detection_artists), nrofareas)
    fig.canvas.draw()

def loadanddraw(imgindex):
    removedetections()
    loadimg(imagefiles[imgindex])
    drawdetections(imagefiles[imgindex])


# EVENT FUNCTIONS

def onkeypress(event):
    global imgindex
    keysteps = {
        'r': 0,
        'right': 1,
        'left': -1,
        'pagedown': 10,
        'pageup': -10,
        'f12': 50,
        'f9': -50,
        'end': len(imagefiles),
        'home': - len(imagefiles)
        }
    if event.key in keysteps:
        step = keysteps[event.key]
    else:
        return
    imgindex += step
    # Enforce 0 <= imgindex < len(imagefiles)
    imgindex = sorted([0, imgindex, len(imagefiles) - 1])[1]
    print "Image index: {}".format(imgindex)
    loadanddraw(imgindex)

def toggle_das(visible):
    # Weird workaround b/c axes_enter_event and axes_leave_event have
    # event.name set to "motion_notify_event"?
    global detection_artists, fig, ax
    for da in detection_artists:
        da.set_visible(visible)
    print "Redrawing"
    fig.canvas.draw()
    print "Done"

fig.canvas.mpl_connect("key_press_event", onkeypress)
# Too slow :/
#fig.canvas.mpl_connect("axes_enter_event", lambda event: toggle_das(False))
#fig.canvas.mpl_connect("axes_leave_event", lambda event: toggle_das(True))


# PLOT AND WAIT FOR USER INPUT

imgindex = 0
detection_artists = []
loadanddraw(imgindex)
plt.show()

"""
for i in range(len(imagefiles)):
    print "{} / {}: {}".format(i+1, len(imagefiles), os.path.basename(imagefiles[i]))
    loadanddraw(i)
    plt.savefig(os.path.join("./testdata/results", os.path.basename(imagefiles[i])), dpi = 300)
    plt.close()
    fig = plt.figure()
    detection_artists = []
    fig.subplots_adjust(.01, .01, .99, .99)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
"""
    

