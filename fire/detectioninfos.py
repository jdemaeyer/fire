"""
FIRE components that describe detected features (e.g. circles).

Each detection info must have a __unicode__() method that returns a string to
be used when a list of detected features should be written out (e.g. the center
and radius of a circle). It should further have a pixels() method that returns
a list of all pixels touched by the detected feature (e.g. all pixels that are
part of the circle). Some processors may use this to no longer detect within
these pixels.
"""

import numpy as np
from fire.base import DetectionInfo
from fire.logging import logger


class DiskInfo(DetectionInfo):

    def __init__(self, x, y, radius):
        # XXX: I think I messed up here, shouldn't it be y first?
        self.x = x
        self.y = y
        self.radius = radius

    def __unicode__(self):
        # XXX: y is first here
        return "d\t{}\t{}\t{}\t{}".format(self.area(), self.y, self.x, self.radius)

    def pixels(self):
        diskpixels = np.zeros((self.y+self.radius, self.x+self.radius), dtype = np.bool)
        return self.markpixels(diskpixels, True)

    def markpixels_old(self, img, value):
        ymax, xmax = img.shape
        xs = np.arange(xmax)
        ys = np.arange(ymax)
        img[(ys[:, None] - newy)**2 + (xs[None, :] - newx)**2
                                            <= self.radius**2] = value
        return img

    def markpixels(self, img, value, ignorebordering = False):
        yshift = np.int(np.round(self.y))
        xshift = np.int(np.round(self.x))
        boxwidth = np.int(np.round(2*np.ceil(self.radius) + 1))
        newy = self.y - yshift + (boxwidth-1) / 2.
        newx = self.x - xshift + (boxwidth-1) / 2.
        yy,xx = np.mgrid[:boxwidth, :boxwidth]
        circle = (yy-newy)**2 + (xx-newx)**2 <= self.radius**2
        circley, circlex = np.where(circle)
        try:
            img[circley+yshift-(boxwidth-1)/2, circlex+xshift-(boxwidth-1)/2] = value
        except IndexError:
            if not ignorebordering:
                # XXX: Need method to kick out out-of-bounds indices
                # Or: Do them one-by-one, e.g. (untested!)
                # for y, x in zip(circley+yshift-(boxwidth-1)/2, circlex+xshift-(boxwidth-1)/2):
                #     try:
                #         img[y, x] = value
                #     except IndexError:
                #         pass
                raise
        return img

    def area(self):
        return np.pi * self.radius**2


class FreeAreaInfo(DetectionInfo):

    def __init__(self, areapixels, background = False):
        self.areapixels = areapixels
        self.background = background

    def __unicode__(self):
        if self.background:
            return "bgfa\t{}".format(self.area())
        else:
            return "fa\t{}".format(self.area())

    def pixels(self):
        return self.areapixels

    def save(self, filename):
        if self.background:
            np.savez_compressed(filename, bgfa = self.areapixels)
        else:
            np.savez_compressed(filename, fa = self.areapixels)

    @classmethod
    def load(cls, filename):
        with np.load(filename) as f:
            for key in ( 'fa', 'bgfa' ):
                try:
                    return cls(f[key], background = (key == 'bgfa'))
                except KeyError:
                    pass
        # No key matched
        raise KeyError


class NonOverlappingFreeAreasInfo(DetectionInfo):

    def __init__(self, labeledareas):
        self.labeledareas = labeledareas

    def addarea(self, areapixels):
        self.labeledareas[areapixels] = np.max(self.labeledareas) + 1

    def pixels(self):
        return self.labeledareas > 0

    def __unicode__(self):
        areastrs = []
        for i, area in zip(*np.unique(self.labeledareas,
                            return_counts = True))[1:]:  # Skip bg (0)
            areastrs.append("nofa\t{}".format(area))
        return "\n".join(areastrs)

    def save(self, filename):
        np.savez_compressed(filename, nofai = self.labeledareas)

    @classmethod
    def load(cls, filename):
        with np.load(filename) as f:
            return cls(f['nofai'])

    @staticmethod
    def smallest_uint_type(maxval):
        for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
            if maxval < np.iinfo(dtype).max:
                return dtype
        # Hopefully never happens, who has an image that fits that many areas?
        logger.error("No numpy uint type can hold {}".format(maxval))
        return np.uint64


class InverseSquareBackgroundInfo(DetectionInfo):

    def __init__(self, imgshape, bbox):
        self.imgshape = imgshape
        self.bbox = bbox

    def pixels(self):
        pixels = np.zeros(self.imgshape, dtype = np.bool)
        ymin, ymax, xmin, xmax = bbox
        pixels[ymin:ymax, xmin:xmax] = True
        # Note the inversion!
        return ~pixels

    def __unicode__(self):
        return ""

