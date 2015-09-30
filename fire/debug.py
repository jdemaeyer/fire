"""
Useful stuff for debugging, often coded quick'n'dirty or copied from random
places, don't use in production...
"""

from fire.logging import logger
import numpy as np
import matplotlib.pyplot as plt


def showimg(img, cbar = False, cmap = plt.cm.Greys_r):
    logger.debug("Plotting image, dimensions: {}".format(img.shape))
    plt.figure()
    plt.imshow(img, interpolation = 'none', cmap = cmap)
    if cbar:
        plt.colorbar()
    plt.show()


def saveimg(filename, img, normalise = True):
    """Save img to filename using colormap cmap."""
    logger.debug("Saving image to {}".format(filename))
    if normalise:
        img -= img.min()
        img /= img.max()
        img = np.uint8(img*255)
    from PIL import Image
    im = Image.fromarray(img)
    im.save(filename)


def randlabels(img):
    maxlabel = img.max()
    newlabels = range(1+int(0.2*maxlabel), 1+int(0.2*maxlabel)+maxlabel+1)
    import random
    random.shuffle(newlabels)
    newlabels[0] = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = newlabels[img[i][j]]


