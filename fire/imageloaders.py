"""
FIRE components to load image information from files. Most of these use lazy
imports to not induce requirements that aren't actually used.
"""

from fire.base import ImageLoader


class MahotasImageLoader(ImageLoader):
    """
    Load images using the mahotas library (which in turn uses a related library
    (imread), freeimage, or pillow (PIL), whichever is available):
        http://luispedro.org/software/mahotas/
    """

    supported_file_types = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'pbm', 'pgm',
                            'ppm', 'tif', 'tiff']

    def __init__(self, **kwargs):
        import mahotas as mh
        self.mh = mh

    def load(self, filename):
        return self.mh.imread(filename)


class RawTherapeeImageLoader(ImageLoader):
    """
    Load RAW images by developing them with Rawtherapee:
        http://rawtherapee.com/
    """
    # TODO
    '''
    def loadnef(self, filename, profile = "rawprofile.pp3", crop = None,
            as_grey = True):
        """Demosaic NEF file using rawtherapee and return numpy array."""
        # Generate temporary file name that's not already taken
        while True:
            tempfile = ".dfinder_temp_{}.png".format(np.random.randint(65536))
            if not os.path.isfile(tempfile):
                break
        # Convert NEF -> TIFF
        command = ["rawtherapee", "-n", "-o", tempfile, "-p", profile,
                "-c", filename]
        subprocess.call(command)
        # Load temporary file into nparray and delete it
        img = self.loadimg(tempfile, crop, as_grey)
        os.remove(tempfile)
        return img
    '''

    def __init__(self):
        raise NotImplementedError

