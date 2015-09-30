"""
FIRE components to process images, i.e. extract detection infos from them.

Every processor must have a __call__() method that accepts two positional
arguments (the image, and the list of lists of previously detected structures)
and returns a list of detection infos. The method may accept further keyword
arguments.
"""

import os.path
import numpy as np
from fire.logging import logger
from fire.base import Processor
from fire.detectioninfos import (DiskInfo, FreeAreaInfo,
        NonOverlappingFreeAreasInfo, InverseSquareBackgroundInfo)
from fire.preprocessors import GaussianPreprocessor, SobelPreprocessor


class DummyProcessor(Processor):

    def __init__(self, **kwargs):
        logger.info("DummyProcessor initialised with keyword args: {}".format(
                    str(kwargs)))

    def reinit(self):
        logger.info("DummyProcessor reinit")

    def __call__(self, img, prev_detected, **kwargs):
        logger.info("DummyProcessor called, returning empty list")
        return []


class FCDProcessor(Processor):
    """
    Implements Fast Circle Detection, see Rad et al.: Fast Circle Detection
    Using Gradient Pair Vectors (2003).
    """

    PREPROCESS = 0
    GRADIENT = 1
    FINDCANDIDATES = 2
    CLUSTER = 3

    def __init__(self, **kwargs):
        from sklearn.cluster import DBSCAN
        self.DBSCAN = DBSCAN
        from scipy import weave
        self.weave = weave
        from scipy.spatial.distance import cdist
        self.cdist = cdist
        from scipy.ndimage.filters import generic_filter
        self.generic_filter = generic_filter

    def reinit(self):
        self.p = {}
        self.img = None
        self.neighbourhood_max_img = None
        self.prev_detected = None
        self.img_preproc = None
        self.grad = None
        self.candidates = None
        self.allcircles = np.zeros((0, 6))
        self.paramindex = 0

    #@profile
    def __call__(self, img, prev_detected, **params):
        """Get list of circles in image using FCD.

        Args:
            img: Image to process (as nparray)
            gaussian: standard deviation for Gaussian kernel
            sobel: boolean value specifying whether to apply sobel
                filter
            minnorm: Gradient norm cutoff
            alpha: FCD gradient angle difference tolerance
            beta: FCD connecting line angle difference tolerance
            gamma: FCD relative gradient norm difference tolerance
            minr: Lower radius cutoff
            maxr: Upper radius cutoff
            radiusscaler: DBSCAN preprocessing radius coefficient
            minmembers: Minimum number of cluster members PER UNIT
                RADIUS for a cluster to be considered a detected circle
            epsilon: DBSCAN neighbourhood size
            minsamples: DBSCAN minimum cluster density in neighbourhood
            maxangspread: TODO DOCUMENT THIS

        Returns:
            A list of DiskInfos
        """
        # Copy given parameters and check for completeness
        self.p.update(params)
        if not self.all_params_set(params):
            logger.critical("First FCD parameter set must be complete")
            return []
        # Figure out if we can use some previous calculations
        startat = self.where_to_start(params)
        logger.debug("Starting at {}".format(startat))
        if self.img is None:
            self.img = np.copy(img)
        if self.neighbourhood_max_img is None and self.p['mincenterlevel'] is not None:
            self.neighbourhood_max_img = self.generic_filter(self.img, np.max, 3)
        if self.prev_detected is None:
            self.prev_detected = list(prev_detected)  # copy
        # Process
        firstnewcircle = len(self.allcircles)
        if startat <= self.PREPROCESS:
            logger.debug("Preprocessing image...")
            self.img_preproc = self.preprocess(
                            self.img, self.p['gaussian'], self.p['sobel'])
        if startat <= self.GRADIENT:
            logger.debug("Computing gradient...")
            self.grad = self.gradient(self.img_preproc, prev_detected)
        if startat <= self.FINDCANDIDATES:
            logger.debug("Finding circle candidates...")
            self.candidates = self.findcandidates(
                    self.grad, self.p['alpha'], self.p['beta'], self.p['gamma'], self.p['minnorm'],
                    self.p['maxr'], self.p['mincenterlevel']
                    )
            logger.debug("Number of candidates: {}".format(
                                                        len(self.candidates)))
        if startat <= self.CLUSTER:
            logger.debug("Clustering...")
            circles = self.cluster(
                    self.candidates, self.p['minr'], self.p['maxr'],
                    self.p['radiusscaler'], self.p['minmembers'],
                    self.p['epsilon'], self.p['minsamples'],
                    self.p['maxangspread']
                    )
            logger.debug("Number of detected circles: {}".format(len(circles)))
            # Shift circles by current offset and append index of this
            # parameter set to all circles
            if len(circles) > 0:
                newcircles = np.hstack((
                            circles,
                            self.paramindex * np.ones((circles.shape[0], 1))
                            ))
                self.allcircles = np.append(
                        self.allcircles,
                        newcircles,
                        axis = 0
                        )
            # OLD:
            #   All detected circles, EXCEPT THE ONES JUST DETECTED IN THIS
            #   RUN, should be removed for future calls to findcandidates
            # NOW:
            #   All detected circles are removed from the gradient map for
            #   future calls to findcandidates
            #return allcircles
            logger.debug("Cleaning circle list...")
            self.allcircles = self.cleancirclelist(self.allcircles)
            logger.debug("Number of new circles: {}".format(len(self.allcircles)
                                                        - firstnewcircle))
            self.paramindex += 1
        return [ DiskInfo(c[0], c[1], c[2])
                for c in self.allcircles[firstnewcircle:] ]

    def all_params_set(self, params):
        return all((x in self.p) for x in ("gaussian", "sobel", "minnorm",
                                      "alpha", "beta", "gamma", "minr", "maxr",
                                      "radiusscaler", "minmembers", "epsilon",
                                      "minsamples", "maxangspread",
                                      "mincenterlevel"))

    def where_to_start(self, params):
        if any((x in params) for x in ("gaussian", "sobel")):
            return self.PREPROCESS
        elif any((x in params) for x in ("minnorm", "alpha", "beta",
                                "gamma", "mincenterlevel")):
            return self.GRADIENT
        elif any((x in params) for x in ("minr", "maxr", "radiusscaler",
                                 "minmembers", "epsilon", "minsamples")):
            return self.CLUSTER
        else:
            return self.PREPROCESS

    def calc_removearea(self, img, prev_detected = None):
        removearea = np.zeros(img.shape, dtype = np.bool)
        if prev_detected is not None:
            for procstep_detections in prev_detected:
                for detected in procstep_detections:
                    removearea[np.where(detected.pixels())] = True
        return removearea

    #@profile
    def preprocess(self, img, gaussian = 3., sobel = False):
        """Preprocess an image with Gaussian and Sobel filters.

        Args:
            See documentation of __call__().

        Returns:
            Preprocessed image as numpy ndarray.
        """
        if gaussian is not None:
            if not hasattr(self, 'gaussianpreproc'):
                self.gaussianpreproc = GaussianPreprocessor()
            img = self.gaussianpreproc(img, sigma = gaussian)
        if sobel:
            if not hasattr(self, 'sobelpreproc'):
                self.sobelpreproc = SobelPreprocessor()
            img = self.sobelpreproc(img)
        return img

    #@profile
    def gradient(self, img, prev_detected, removearea = True):
        """Return numpy array of gradient vectors from a given image,
        sorted by direction.

        Args:
            See documentation of __call__().

        Returns:
            An nparray of gradient vectors, each entry being an nparray
            with format:
                [y, x, dy, dx, norm(dx, dy), arg(dx, dy)]
        """
        rawgrad = np.gradient(img)
        if removearea:
            ra = self.calc_removearea(img, prev_detected)
            rawgrad[0][ra] = 0.
            rawgrad[1][ra] = 0.
        rawgrid = np.meshgrid(
            np.arange(img.shape[0]),
            np.arange(img.shape[1]),
            indexing = 'xy')
        # TODO: Can we get rid of the reshape?
        grad = np.hstack((
            tuple( x.reshape((-1,1)) for x in
                (rawgrid[0].T, rawgrid[1].T, rawgrad[0], rawgrad[1]) )
            + (
                np.zeros((img.shape[0]*img.shape[1], 2)),
            )
            ))
        grad[:,4] = np.linalg.norm(grad[:,2:4], axis = 1)
        # NOTE: We could save some arctan2 calculations if we moved the
        #       "norm > minnorm" statement here, but the composite FCD
        #       might wish to use different minnorms with the same pre-
        #       processing parameters, so we put it into the
        #       findcandidates function
        # Sort vectors by direction
        grad[:,5] = np.arctan2(grad[:,2], grad[:,3])
        grad = grad[grad[:,5].argsort()]
        return grad

    #@profile
    def findcandidates(self, grad, alpha, beta, gamma, minnorm = 0.,
                maxr = None, mincenterlevel = None):
        """Return numpy array of FCD circle candidates from a list of
        vectors.

        Args:
            See documentation of __call__().

        Returns:
            An nparray of circle candidates, each entry being an nparray
            with format:
                [C_x, C_y, r]
        """
        if maxr is None:
            maxr = max(self.img.shape)
        # Remove all vectors whose magnitude is on the order of noise
        grad = grad[grad[:,4] > minnorm]
        # Maximum number of candidates is len(grad) * (len(grad)-1) / 2
        # (one candidate for every possible pair), but this should be
        # sufficient:
        candidates = np.zeros((len(grad) * 20, 4), dtype=np.float64)
        # Load C code and needed variables
        with open(os.path.join(os.path.dirname(__file__), "cpp/fcd.c"), 'r'
                ) as ccodefile:
            ccode = ccodefile.read()
        y, x, dy, dx, norms, angles = np.hsplit(grad, 6)
        nrofgrads = len(grad)
        # Compute candidates with compiled C code
        nrofcandidates = self.weave.inline(
                ccode,
                ['x', 'y', 'dx', 'dy', 'norms', 'angles', 'nrofgrads',
                 'candidates', 'alpha', 'beta', 'gamma', 'maxr'],
                headers = ["<cmath>"],
                type_converters = self.weave.converters.blitz,
                compiler = 'gcc')
        # Cut candidates vector to actual number of candidates
        candidates = candidates[0:nrofcandidates]
        # Filter candidates for minimum grey level at center
        if mincenterlevel is not None:
            removelist = []
            for i, c in enumerate(candidates):
                if self.neighbourhood_max_img[c[1], c[0]] < mincenterlevel:
                    removelist.append(i)
            candidates = np.delete(candidates, removelist, axis = 0)
        return candidates

    #@profile
    def cluster(self, candidates, minr, maxr, radiusscaler,
                     minmembers, epsilon, minsamples, maxangspread):
        """Return detected circles by applying DBSCAN to cluster
        candidates.

        Args:
            See documentation of __call__().

        Returns:
            An nparray of detected circles, each entry being an nparray
            with format:
                [C_x, C_y, r]
        """
        if minr is not None:
            candidates = candidates[candidates[:,2] >= minr]
        if maxr is not None:
            candidates = candidates[candidates[:,2] <= maxr]
        # Preprocess radius, cluster, and return radius to original state
        if radiusscaler is not None:
            candidates[:,2] *= radiusscaler
        try:
            db = self.DBSCAN(eps = epsilon, min_samples = minsamples).fit(candidates[:,:3])
        except ValueError:
            # No clusters found
            return np.zeros((0, 5))
        finally:
            if radiusscaler is not None:
                candidates[:,2] /= radiusscaler
        labels = db.labels_
        unique_labels = set(db.labels_)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # Get center of masses for each cluster
        circles = np.zeros((n_clusters, 5))
        nrofcircles = 0
        for L in unique_labels:
            if L < 0: continue # noise
            members = [index[0] for index in np.argwhere(labels == L)]
            circle = np.mean(candidates[members][:,:3], axis = 0)
            # Minimum number of members?
            if minmembers is not None and len(members) < minmembers*np.sqrt(circle[2]):
                continue
            # Angle spread okay?
            # Get spread of diametrically bimodal angle distribution
            # http://webspace.ship.edu/pgmarr/Geo441/Lectures/Lec%2016%20-%20Directional%20Statistics.pdf
            angles = np.fmod(candidates[members][:,3]*2, 2*np.pi)
            r_sq = np.mean(np.sin(angles))**2 + np.mean(np.cos(angles))**2
            if r_sq > maxangspread**2:
                continue
            # Circle okay!
            circles[nrofcircles][:3] = circle
            circles[nrofcircles][3] = len(members)
            circles[nrofcircles][4] = r_sq
            nrofcircles += 1
        circles = circles[:nrofcircles]
        return circles

    #@profile
    def cleancirclelist(self, circles):
        """Return list containing no overlapping circles."""
        # Sort circles by radius, largest to smallest
        #circles = circles[circles[:,2].argsort()[::-1]]
        removelist = []
        # TODO: Might want to rewrite below using pdist (sparse distance matrix)
        # https://stackoverflow.com/questions/5323818/condensed-matrix-function-to-find-pairs/14839010#14839010
        dists = self.cdist(circles[:,:2], circles[:,:2])
        # Get boolean matrix of circle pairs where distance is smaller than 10%
        # of radius of first circle
        mergecandidates = dists < .1 * circles[:,2]
        # Mirror along diagonal with logical or
        mergecandidates = np.logical_or(mergecandidates, mergecandidates.T)
        # Erase lower triangle and diagonal
        mergecandidates = np.triu(mergecandidates, 1)
        # Convert to indices
        mergecandidates = np.transpose(np.where(mergecandidates))
        # Get these where radii are roughly (< 10% difference) the same
        mergecandidates = mergecandidates[ np.where(
                circles[mergecandidates[:,0],2] / circles[mergecandidates[:,1],2] > 0.90
                ) ]
        mergecandidates = mergecandidates[ np.where(
                circles[mergecandidates[:,0],2] / circles[mergecandidates[:,1],2] < 1.11
                ) ]
        logger.debug("{} mergers".format(len(mergecandidates)))
        self.totmergers += len(mergecandidates)
        # Merge remaining candidates
        for c1_index, c2_index in mergecandidates:
            c1, c2 = circles[c1_index], circles[c2_index]
            c1[:3] = (c1[:3] * c1[3] + c2[:3] * c2[3])/(c1[3] + c2[3])
            if dists[c1_index, c2_index] > 0.5:
                c1[3] += c2[3]
            else:
                # Probably duplicates from stitching our tiles
                c1[3] = max(c1[3], c2[3])
            c1[4] = min(c1[4], c2[4])
            removelist.append(c2_index)
        circles = np.delete(circles, removelist, axis = 0)
        dists = np.delete(dists, removelist, axis = 0)
        dists = np.delete(dists, removelist, axis = 1)
        removelist = []
        # Similar to above, except now get all circles that overlap
        overlaplist = dists - circles[:,2] - np.matrix(circles[:,2]).T < -1
        overlaplist = np.triu(overlaplist, 1)
        overlaplist = np.transpose(np.where(overlaplist))
        logger.debug("{} overlaps".format(len(overlaplist)))
        self.totoverl += len(overlaplist)
        # Count number of occurences for each circle (no matter if c1 or c2)
        counts = np.bincount(overlaplist.flat)
        # Remove all circles that have two or more overlaps
        while len(counts) > 0 and counts.max() > 1:
            badcircle = counts.argmax()
            removelist.append(badcircle)
            # Remove all rows from overlaplist that contain badcircle
            # http://stackoverflow.com/questions/11453141/
            overlaplist = overlaplist[~(overlaplist == badcircle).any(axis=1)]
            counts = np.bincount(overlaplist.flat)
        # All remaining pairs are overlaps of only two circles
        for c1_index, c2_index in overlaplist:
            c1, c2 = circles[c1_index], circles[c2_index]
            # Remove circle with fewer members
            if c1[3] > c2[3]:
                removelist.append(c2_index)
            else:
                removelist.append(c1_index)
        return np.delete(circles, removelist, axis = 0)


class TiledFCDProcessor(FCDProcessor):

    #@profile
    def __call__(self, img, prev_detected, **params):
        if 'tiles' in params:
            params['tiles_v'] = params['tiles']
            params['tiles_h'] = params['tiles']
        elif not ('tiles_v' in params and 'tiles_h' in params):
            logger.error(("Invalid tiling information given to "
                "TiledFCDProcessor. Provide either 'tiles' or both 'tiles_h' "
                "and 'tiles_v'."))
            return []
        # Copy given parameters and check for completeness
        self.p.update(params)
        if not self.all_params_set(params):
            logger.critical("First FCD parameter set must be complete")
            return []
        if self.img is None:
            self.img = np.copy(img)
        if self.neighbourhood_max_img is None and self.p['mincenterlevel'] is not None:
            self.neighbourhood_max_img = self.generic_filter(self.img, np.max, 3)
        if self.prev_detected is None:
            self.prev_detected = list(prev_detected)  # copy
        # Process
        firstnewcircle = len(self.allcircles)
        logger.debug("Preprocessing image...")
        #try:
        #    import pickle
        #    gradcircles = pickle.load(open("testdata/circles.pickle_5tiles", 'r'))
        #except IOError:
        if True:
            img_preproc = self.preprocess(
                            self.img, self.p['gaussian'], self.p['sobel'])
            logger.debug("Computing gradient...")
            grads = self.tiled_gradients(img_preproc, prev_detected,
                            self.p['tiles_v'], self.p['tiles_h'], self.p['maxr']+3)
            gradcircles = []
            for i, grad in enumerate(grads):
                logger.debug("Tile {} / {}".format(i+1, len(grads)))
                logger.debug("Finding circle candidates...")
                candidates = self.findcandidates(
                        grad, self.p['alpha'], self.p['beta'], self.p['gamma'],
                        self.p['minnorm'], self.p['maxr'], self.p['mincenterlevel']
                        )
                logger.debug("Number of candidates: {}".format(
                                                            len(candidates)))
                logger.debug("Clustering...")
                circles = self.cluster(
                        candidates, self.p['minr'], self.p['maxr'],
                        self.p['radiusscaler'], self.p['minmembers'],
                        self.p['epsilon'], self.p['minsamples'],
                        self.p['maxangspread']
                        )
                logger.debug("Number of detected circles: {}".format(len(circles)))
                # Attach index of this tile to circles
                circles = np.hstack((
                    circles,
                    i * np.ones((circles.shape[0], 1))
                    ))
                gradcircles.append(circles)
            #import pickle
            #pickle.dump(gradcircles, open("testdata/circles.pickle", 'w'), -1)
        logger.debug("Cleaning circle list...")
        self.totmergers = 0
        self.totoverl = 0
        for i in range(self.p['tiles_v'] - 1):
            for j in range(self.p['tiles_h'] - 1):
                logger.debug("Circle group {} / {}".format(i * (self.p['tiles_h']-1) + j + 1, (self.p['tiles_v']-1)*(self.p['tiles_h']-1)))
                tileindices = (  i*self.p['tiles_h'] + j,      i*self.p['tiles_h'] + j+1,
                                (i+1)*self.p['tiles_h'] + j,  (i+1)*self.p['tiles_h'] + j+1 )
                # Put circles of four-tile-groups into same list and clean
                groupcircles = np.concatenate(
                            tuple( gradcircles[ti] for ti in tileindices ) )
                groupcircles = self.cleancirclelist(groupcircles)
                # Split returned list back into corresponding gradcircles
                for x in tileindices:
                    gradcircles[x] = groupcircles[ groupcircles[:,-1] == x ]
        logger.debug("Total mergers: {}".format(self.totmergers))
        logger.debug("Total overlaps: {}".format(self.totoverl))
        newcircles = np.concatenate( gradcircles )
        logger.debug("Number of new circles: {}".format(len(newcircles)))
        self.allcircles = np.concatenate( (self.allcircles, newcircles) )
        return [ DiskInfo(c[0], c[1], c[2])
                for c in self.allcircles[firstnewcircle:] ]

    #@profile
    def tiled_gradients(self, img, prev_detected, tiles_v, tiles_h, overlap,
            removearea = True):
        rawgrad = np.gradient(img)
        if removearea:
            ra = self.calc_removearea(img, prev_detected)
            rawgrad[0][ra] = 0.
            rawgrad[1][ra] = 0.
        rawgrid = np.meshgrid(
            np.arange(img.shape[0]),
            np.arange(img.shape[1]),
            indexing = 'xy')
        # Get tile edge indices
        ymins = np.linspace(0, img.shape[0], tiles_v+1).astype(int)[:tiles_v]
        ymaxs = np.clip( np.linspace(0, img.shape[0], tiles_v+1).astype(int)[1:] + overlap,  0,  img.shape[0] )
        xmins = np.linspace(0, img.shape[1], tiles_h+1).astype(int)[:tiles_h]
        xmaxs = np.clip( np.linspace(0, img.shape[1], tiles_h+1).astype(int)[1:] + overlap,  0,  img.shape[1] )
        grads = []
        for ymin, ymax in zip(ymins, ymaxs):
            for xmin, xmax in zip(xmins, xmaxs):
                thisgrad = np.hstack((
                    tuple( x.reshape((-1,1)) for x in
                        (rawgrid[0][xmin:xmax, ymin:ymax].T,
                         rawgrid[1][xmin:xmax, ymin:ymax].T,
                         rawgrad[0][ymin:ymax, xmin:xmax],
                         rawgrad[1][ymin:ymax, xmin:xmax]) )
                    + (
                        np.zeros(((ymax - ymin)*(xmax - xmin), 2)),
                    )
                    ))
                thisgrad[:,4] = np.linalg.norm(thisgrad[:,2:4], axis = 1)
                thisgrad[:,5] = np.arctan2(thisgrad[:,2], thisgrad[:,3])
                thisgrad = thisgrad[thisgrad[:,5].argsort()]
                grads.append(thisgrad)
        return grads


class DiskExtendProcessor(Processor):

    def __init__(self, **kwargs):
        angles = np.linspace(0, 2*np.pi, 100)
        self.sins = np.sin(angles)
        self.coss = np.cos(angles)
        from scipy.signal import argrelextrema
        self.argrelextrema = argrelextrema

    #@profile
    def __call__(self, img, prev_detected, dr_min = 0., dr_max = 10.,
                 dr_steps = 100):
        drs = np.linspace(dr_min, dr_max, dr_steps)
        for procd in prev_detected:
            for d in procd:
                if not isinstance(d, DiskInfo):
                    continue
                sums = np.zeros(dr_steps, dtype=float)
                for i, dr in enumerate(drs):
                    r = d.radius + dr
                    circle_ys = np.round(d.y + self.sins*r).astype(np.int)
                    circle_xs = np.round(d.x + self.coss*r).astype(np.int)
                    try:
                        sums[i] = np.sum(img[circle_ys, circle_xs])
                    except IndexError:
                        # Reached image boundary
                        break
                minindices = self.argrelextrema(sums, np.less_equal, order = 5)
                try:
                    firstminindex = minindices[0][0]
                except IndexError:
                    # No minimum, keep original radius
                    continue
                d.radius = d.radius + drs[firstminindex]
        return []


class RemoveOuterProcessor(Processor):

    def __init__(self, **kwargs):
        import mahotas as mh
        self.mh = mh

    def __call__(self, img, prev_detected, padding = 0):
        binary = np.zeros(img.shape, dtype = np.bool)
        for pdetected in prev_detected:
            for d in pdetected:
                binary[d.pixels()] = True
        binary = self.mh.polygon.fill_convexhull(binary)
        for i in range(padding):
            binary = self.mh.dilate(binary)
        binary = ~binary
        return [ FreeAreaInfo(binary, background = True) ]

        #return [ InverseSquareBackgroundInfo(
        #            img.shape,
        #            self.get_bbox(binary, padding)
        #        ) ]

    def get_bbox(self, binary, extend = 0):
        ymin, ymax, xmin, xmax = self.mh.bbox(binary)
        ymin = max(0, ymin - extend)
        ymax = min(binary.shape[0], ymax + extend)
        xmin = max(0, xmin - extend)
        xmax = min(binary.shape[1], xmax + extend)
        return ymin, ymax, xmin, xmax

#class NOFAIExtendProcessor(Processor):
#
#    def __call__(self, img, prev_detected, max_dilation = 10):
#        for procd in prev_detected:
#            for d in procd:
#                if not isinstance(d, NonOverlappingFreeAreasInfo):
#                    continue


class WatershedProcessor(Processor):

    def __init__(self, **kwargs):
        import mahotas as mh
        self.mh = mh
        from scipy.ndimage import binary_fill_holes
        self.binary_fill_holes = binary_fill_holes

    def __call__(self, img, prev_detected, seed_threshold = None,
                 erode_factor = 1, minsize = 0, maxhole = 0.5, checkQsize = 0,
                 minQ = 0.5,
                 **kwargs):
        # Guess threshold if not given
        if seed_threshold is None:
            from fire.debug import showimg
            # Otsu needs int, convert to range 0-255
            normimg = (img - img.min()) / (img.max() - img.min())
            intimg = np.rint(normimg*255).astype(np.uint8)
            intimg_threshold = self.mh.thresholding.otsu(intimg)
            # Convert received threshold back to image range
            seed_threshold = (intimg_threshold / 255. * (img.max() - img.min())
                              + img.min())
            logger.debug("Guessed threshold (Otsu): {}".format(seed_threshold))
        # Find seeds
        binary = img > seed_threshold
        for i in range(erode_factor):
            binary = self.mh.morph.erode(binary)
        from fire.debug import showimg
        showimg(binary)
        seeds,nr_of_seeds = self.mh.label(binary)
        logger.debug("Number of watershed seeds: {}".format(nr_of_seeds))
        # Do watershed transformation
        labeled = self.mh.cwatershed(img.max() - img, seeds)
        logger.debug("Number of areas: {}".format(len(labeled)))
        # Extract sizes of area and filter for minsize
        sizes = np.bincount(labeled.flat)
        bigareas = np.argwhere(sizes > maxhole*minsize)
        logger.debug("With minimum size (holes not filled) {}: {}".format(
                                            maxhole*minsize, len(bigareas)))
        # Create detection info
        detected = NonOverlappingFreeAreasInfo(np.zeros(img.shape, dtype =
                NonOverlappingFreeAreasInfo.smallest_uint_type(len(bigareas))))
        for bigarea in bigareas:
            filled = self.fill_holes(labeled == bigarea)
            size = np.count_nonzero(filled)
            if size < minsize:
                continue
            elif size < checkQsize and self.isoperimetric_quotient(filled) < minQ:
                continue
            else:
                detected.addarea(filled)
        logger.debug("Found {} areas".format(np.max(detected.labeledareas)))
        return [detected]

    def fill_holes(self, binary):
        ymin, ymax, xmin, xmax = self.mh.bbox(binary)
        binary[ymin:ymax, xmin:xmax] = self.binary_fill_holes(
                                                binary[ymin:ymax, xmin:xmax])
        return binary
        # Alternative, faster?:
        # 1. Label labeled == bigarea
        # 2. Get "new" label of our bigarea
        # 3. Identify background: Biggest or second biggest area
        # 4. All other labels are holes
        # See also: https://stackoverflow.com/questions/14385921/

    def isoperimetric_quotient(self, binary):
        ymin, ymax, xmin, xmax = self.mh.bbox(binary)
        perimeter = np.count_nonzero(self.mh.bwperim(binary[ymin:ymax, xmin:xmax]))
        return 4 * np.pi * np.count_nonzero(binary[ymin:ymax, xmin:xmax]) / perimeter**2


class DirtyRemoveEdgesProcessor(Processor):

    def __call__(self, img, prev_detected, **kwargs):
        for procd in prev_detected:
            for d in procd:
                if not isinstance(d, NonOverlappingFreeAreasInfo):
                    break
                d.labeledareas[d.labeledareas == d.labeledareas[1,1]] = 0
                d.labeledareas[d.labeledareas == d.labeledareas[1,-1]] = 0
                d.labeledareas[d.labeledareas == d.labeledareas[-1,1]] = 0
                d.labeledareas[d.labeledareas == d.labeledareas[-1,-1]] = 0
        return []


class ThresholdProcessor(Processor):

    class RejectArea(Exception):
        pass

    def __init__(self, **kwargs):
        import mahotas as mh
        self.mh = mh
        from scipy.ndimage import binary_fill_holes
        self.binary_fill_holes = binary_fill_holes

    #@profile
    def __call__(self, img, prev_detected, threshold = None, minsize = 0,
                 maxsize = None, steps_prelabel = [], steps_postlabel = [],
                 **kwargs):
        if threshold is None:
            threshold = self.otsu(img)
            logger.debug("Guessed threshold (Otsu): {}".format(threshold))
        binary = img > threshold
        for pdetected in prev_detected:
            for d in pdetected:
                binary[d.pixels()] = False
        self.backgroundinfos = []
        binary = self.do_steps(img, binary, steps_prelabel)
        # Label
        labeled, nr_labels = self.mh.label(binary)
        logger.debug("Number of areas: {}".format(nr_labels))
        # Filter for size requirements
        sizes = np.bincount(labeled.flat)
        considered_sizes = sizes >= minsize
        if maxsize is not None:
            considered_sizes = np.logical_and(
                                    considered_sizes, sizes <= maxsize)
        considered_labels = np.argwhere(considered_sizes)
        logger.debug("Meeting size requirements: {}".format(
                                                    len(considered_labels)))
        # Create detection info
        detected = NonOverlappingFreeAreasInfo(
                        np.zeros(img.shape,
                                 dtype =
                                 NonOverlappingFreeAreasInfo.smallest_uint_type(
                                    len(considered_labels))))
        for area_label in considered_labels:
            if area_label == 0:
                continue  # ignore background
            thisarea = (labeled == area_label)
            try:
                thisarea = self.do_steps(img, thisarea, steps_postlabel)
            except self.RejectArea:
                continue
            detected.addarea(thisarea)
        logger.debug("Found {} areas".format(np.max(detected.labeledareas)))
        return self.backgroundinfos + [detected]

    def get_steps(self, steplist):
        steps = []
        for step in steplist:
            try:
                method, args = step.items()[0]
            except AttributeError:
                # step is not a dict (no args)
                method = step
                args = []
            if not isinstance(args, list):
                args = [ args ]
            # Does method exist?
            if not hasattr(self, method):
                logger.error("Unknown method: '{}', ignoring".format(method))
                continue
            steps.append( (method, args) )
        return steps

    def do_steps(self, img, binary, steplist):
        for method, args in self.get_steps(steplist):
            binary = getattr(self, method)(img, binary, *args)
        return binary

    def otsu(self, img):
        # Otsu needs int, convert to range 0-255
        normimg = (img - img.min()) / (img.max() - img.min())
        intimg = np.rint(normimg*255).astype(np.uint8)
        intimg_threshold = self.mh.thresholding.otsu(intimg)
        # Convert received threshold back to image range
        threshold = (intimg_threshold / 255. * (img.max() - img.min())
                          + img.min())

    def get_bbox(self, binary, extend = 0):
        ymin, ymax, xmin, xmax = self.mh.bbox(binary)
        ymin = max(0, ymin - extend)
        ymax = min(binary.shape[0], ymax + extend)
        xmin = max(0, xmin - extend)
        xmax = min(binary.shape[1], xmax + extend)
        return ymin, ymax, xmin, xmax

    def _morph(self, binary, method, steps = 0):
        if steps == 0:
            return binary
        if method not in ( 'erode', 'dilate', 'open', 'close'):
            return binary
        if method == 'erode':
            ext = 1
        elif method == 'open' or method == 'close':
            ext = 2
        elif method == 'dilate':
            ext = 1 + steps
        newbinary = binary.copy()
        ymin, ymax, xmin, xmax = self.get_bbox(binary, extend = ext)
        for i in range(steps):
            newbinary[ymin:ymax, xmin:xmax] = getattr(self.mh, method)(
                    newbinary[ymin:ymax, xmin:xmax])
        return newbinary

    def erode(self, img, binary, steps = 1):
        return self._morph(binary, 'erode', steps)

    def dilate(self, img, binary, steps = 1):
        return self._morph(binary, 'dilate', steps)

    def open(self, img, binary, steps = 1):
        return self._morph(binary, 'open', steps)

    def close(self, img, binary, steps = 1):
        return self._morph(binary, 'close', steps)

    def fill(self, img, binary):
        # TODO: Holes that touch borders are not closed, do we want them to be?
        return self.binary_fill_holes(binary)

    def remove_bordering(self, img, binary, min_touching = 1):
        """Remove areas that touch the border with at least min_touching pixels
        """
        labeled, nr_labels = self.mh.label(binary)
        borderpixels = np.concatenate( (
                            labeled[0, :],
                            labeled[-1, :],
                            labeled[1:-1, 0],
                            labeled[1:-1, -1]
                            ) )
        bordercounts = np.bincount(borderpixels)
        removelabels = np.argwhere(bordercounts >= min_touching).flatten()
        bgarea = np.zeros_like(binary)
        for l in removelabels:
            binary[labeled == l] = False
            bgarea[labeled == l] = True
        #self.backgroundinfos.append(FreeAreaInfo(bgarea, background = True))
        return binary

    def isoperimetric_quotient(self, binary):
        ymin, ymax, xmin, xmax = self.mh.bbox(binary)
        perimeter = np.count_nonzero(self.mh.bwperim(binary[ymin:ymax, xmin:xmax]))
        return 4 * np.pi * np.count_nonzero(binary[ymin:ymax, xmin:xmax]) / perimeter**2

    def minQ(self, img, binary, minq):
        if self.isoperimetric_quotient(binary) < minq:
            raise self.RejectArea
        return binary

    def maxstddev(self, img, binary, maxstd):
        if np.std(img[binary]) > maxstd:
            raise self.RejectArea
        return binary

    def calccornerresponse(self, img, binary):
        # Lazy import to avoid unnecessary dependency if not used
        from skimage.feature import corner_harris
        self.cr = corner_harris(binary)
        return binary

    def maxcornerresponse(self, img, binary, maxcr):
        ymin, ymax, xmin, xmax = self.get_bbox(binary, extend = 1)
        perim = np.zeros(binary.shape, dtype = np.bool)
        perim[ymin:ymax, xmin:xmax] = self.mh.bwperim(
                                                binary[ymin:ymax, xmin:xmax])
        #perimlength = np.count_nonzero(self.mh.bwperim(
        #                            binary[ymin:ymax, xmin:xmax]))

        #area = self.mh.dilate(self.mh.dilate(binary))
        #import matplotlib.pyplot as plt
        #plt.imshow(self.cr[ymin:ymax, xmin:xmax], cmap = 'Greys', interpolation = 'none')
        #plt.imshow(binary[ymin:ymax, xmin:xmax], alpha = 0.3, interpolation = 'none')
        #plt.show()
        # Total corner response on boundary divided by boundary size
        cr = np.sum( self.cr[binary] ) / np.count_nonzero(
                                                perim[ymin:ymax, xmin:xmax])
        if cr > maxcr:
            raise self.RejectArea
        return binary

    def minconvexratio(self, img, binary, minratio):
        ymin, ymax, xmin, xmax = self.get_bbox(binary, extend = 1)
        convexhull = self.mh.polygon.fill_convexhull(binary[ymin:ymax, xmin:xmax])
        convexratio = (np.float(np.count_nonzero(binary[ymin:ymax, xmin:xmax])) /
                        np.count_nonzero(convexhull))
        if convexratio < minratio:
            raise self.RejectArea
        return binary

