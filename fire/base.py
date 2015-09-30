"""
Base classes for common components of FIRE, start developing your own image
loaders, (pre-)processors, detection infos by subclassing from here!
"""

import numpy as np


class Processor(object):

    def __init__(self, **kwargs):
        pass

    def reinit(self):
        pass

    def __call__(self, img, prev_detected, **kwargs):
        raise NotImplementedError


class Preprocessor(object):

    def __init__(self, **kwargs):
        pass

    def __call__(self, img, **kwargs):
        raise NotImplementedError


class ImageLoader(object):

    supported_file_types = []

    def __init__(self, **kwargs):
        pass

    def load(self, filename):
        raise NotImplementedError


class DetectionInfo(object):

    name = "GenericDetectionInfo"

    def __init__(self, **kwargs):
        pass

    def __str__(self):
        return unicode(self)

    def __unicode__(self):
        raise NotImplementedError

    def pixels(self):
        """Pixels occupied by this detection"""
        raise NotImplementedError

    def area(self):
        return np.count_nonzero(self.areapixels)


class Exporter(object):

    def __init__(self, **kwargs):
        pass

    def __call__(self, infile, detectioninfos, **kwargs):
        raise NotImplementedError

    def skip(self, infile):
        return False

