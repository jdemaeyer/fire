"""
FIRE components to preprocess images, e.g. filtering or background removal.

Every preprocessor must have a __call__() method that excepts one positional
argument (the image) and returns an image (numpy ndarray). The method may
accept further keyword arguments.
"""

import numpy as np

import fire.errors
from fire.logging import logger
from fire.base import Preprocessor


class CropPreprocessor(Preprocessor):
    """Crop image to given rectangle or circle"""

    def __call__(self, img, croptype = "rect", **ppargs):
        if croptype == "rect":
            if not all(( z in ppargs for z in ['x', 'y', 'dx', 'dy'] )):
                logger.error(("Crop processor needs x, y, dx, and dy parameters"
                              " for type 'rect'. Doing nothing!"))
                return img
            return img[ ppargs['y'] : ppargs['y'] + ppargs['dy'],
                        ppargs['x'] : ppargs['x'] + ppargs['dx'] ]
        elif croptype == "circular":
            if not all(( z in ppargs for z in ['x', 'y', 'r'] )):
                logger.error(("Crop processor needs x, y, and r parameters for "
                              "type 'circular'. Doing nothing!"))
                return img
            xmin = np.floor(ppargs['x'] - ppargs['r'])
            xmax = np.ceil(ppargs['x'] + ppargs['r']) + 1
            ymin = np.floor(ppargs['y'] - ppargs['r'])
            ymax = np.ceil(ppargs['y'] + ppargs['r']) + 1
            procimg = img[ymin:ymax, xmin:xmax]
            ys = np.arange(ymax - ymin)
            xs = np.arange(xmax - xmin)
            procimg[(ys[:, None] + ymin - ppargs['y'])**2 + (xs[None, :] +
                    xmin - ppargs['x'])**2 > ppargs['r']**2] = procimg.min()
            procimg[(ys[:, None] + ymin - ppargs['y'])**2 + (xs[None, :] +
                    xmin - ppargs['x'])**2 > (ppargs['r']+2)**2] = procimg.max()
            return procimg
        else:
            logger.error("Unknown crop method: '{}', doing nothing!".format(
                            croptype))


class MakeFloatPreprocessor(Preprocessor):
    """Make sure image intensity is given as floating point numbers"""

    def __call__(self, img):
        return img.astype(np.float64)


class GreyscalePreprocessor(Preprocessor):
    """Convert colour image to greyscale"""

    def __init__(self, **kwargs):
        import mahotas as mh
        self.mh = mh

    def __call__(self, img, channel = None):
        if len(img.shape) == 2:
            logger.warning(("Greyscale preprocessor received greyscale image, "
                            "doing nothing."))
            return img
        if channel is None:
            return self.mh.colors.rgb2grey(img)
        else:
            return img[:,:,channel]


class GaussianPreprocessor(Preprocessor):

    def __init__(self, **kwargs):
        import mahotas as mh
        self.mh = mh

    def __call__(self, img, sigma = 3.):
        """sigma: Standard deviation of Gaussian kernel"""
        # FIXME: This is slow. Does mahotas already use this:
        #        http://blog.ivank.net/fastest-gaussian-blur.html ?
        return self.mh.gaussian_filter(img, sigma)


class SobelPreprocessor(Preprocessor):

    def __init__(self, **kwargs):
        import mahotas as mh
        self.mh = mh

    def __call__(self, img, just_filter = True):
        """just_filter: Will not binarise if set to True (default)"""
        try:
            return self.mh.edge.sobel(img, just_filter = just_filter)
        except ValueError:
            logger.error(("Sobel preprocessor can only work on greyscale "
                "images, doing nothing!"))
            return img


class RemoveBackgroundPreprocessor(Preprocessor):

    def __init__(self, manager = None, **kwargs):
        self.manager = manager
        self.bgs = {}
        self.bgs_nonzero = {}

    def __call__(self, img, bgsource = None, methods = ['subtract'], **kwargs):
        if bgsource is None:
            logger.error(("No background image given to "
                    "RemoveBackgroundPreprocessor, doing nothing!"))
            return img
        if bgsource not in self.bgs:
            logger.debug("Loading background image {}".format(bgsource))
            # Load background image
            try:
                self.bgs[bgsource] = self.manager.loadimage(bgsource)
            except IOError, fire.errors.ImageCannotBeLoadedError:
                logger.error(("Unable to load background image '{}', doing "
                             "nothing!").format(bgsource))
                return img
            # If preprocessing steps are not explicity given, use preprocessing
            # chain used for all images, up to the point where background is
            # removed
            if not 'preprocsteps' in kwargs:
                preprocsteps = self.get_previous_preprocsteps()
                if preprocsteps is None:
                    return img
            else:
                preprocsteps = kwargs['preprocsteps']
            self.bgs[bgsource] = self.manager.preprocess(self.bgs[bgsource],
                                                         preprocsteps)
            self.bgs_nonzero[bgsource] = self.bgs[bgsource].copy()
            self.bgs_nonzero[bgsource][self.bgs[bgsource] == 0.] = (0.01 *
                                                    self.bgs[bgsource].max())
            logger.debug("Background image {} loaded".format(bgsource))
        procimg = img.copy()
        for method in methods:
            if method == 'sub':
                procimg -= self.bgs[bgsource]
            elif method == 'div':
                procimg /= self.bgs[bgsource]
            elif method == 'divnorm':
                procimg /= self.bgs_nonzero[bgsource] / self.bgs[bgsource].max()
            elif method == 'divsqrt':
                procimg /= np.sqrt(self.bgs_nonzero[bgsource])
            elif method == 'divsqrtnorm':
                procimg /= np.sqrt(self.bgs_nonzero[bgsource] / self.bgs[bgsource].max())
            elif method == 'div3rdrootnorm':
                procimg /= np.power(self.bgs_nonzero[bgsource] / self.bgs[bgsource].max(), 1./64.)
            elif method == 'signedthirdroot':
                procimg = np.power(np.abs(procimg), 1./3.)*np.sign(procimg)
            else:
                logger.error(("Unknown background removal method: '{}', doing "
                              "nothing!").format(method))
        return procimg

    def get_previous_preprocsteps(self):
        # Find myself in preprocessor list
        mykey = None
        for key, preproc in self.manager.preprocessors.iteritems():
            if preproc is self:
                mykey = key
                break
        if mykey is None:
            logger.error(("RemoveBackgroundProcessor lives outside of "
                          "manager.preprocessors? Doing nothing!"))
            return None
        try:
            mypreprocindex = [ x[0] for x in self.manager.preprocsteps
                             ].index(mykey)
        except ValueError:
            logger.error(("RemoveBackgroundProcessor is not in "
                          "preprocessing chain? Doing nothing!"))
            return None
        return self.manager.preprocsteps[:mypreprocindex]


class CLAHEPreprocessor(Preprocessor):

    def __init__(self, **kwargs):
        from skimage.exposure import equalize_adapthist
        self.equalize_adapthist = equalize_adapthist

    def __call__(self, img, **kwargs):
        try:
            return self.equalize_adapthist(img, **kwargs)
        except ValueError:
            # Float images need to be between 0 and 1 (-1 and 1 according to
            # documentation, but result always seems to be 0 - 1...), normalise
            # to that range
            oldmax = img.max()
            oldmin = img.min()
            oldrange = oldmax - oldmin
            newmax = 1.
            newmin = 0.
            newrange = newmax - newmin
            normimg = ( img - oldmin ) / oldrange * newrange + newmin
            # CLAHE on normalised image, transform back to old range
            normimg = self.equalize_adapthist(normimg, **kwargs)
            return ( normimg - newmin ) / newrange * oldrange + oldmin


