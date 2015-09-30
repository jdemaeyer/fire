import os
from importlib import import_module

from fire.logging import logger
import fire.imageloaders
import fire.preprocessors
import fire.processors
import fire.exporters
import fire.errors


class ProcessingManager(object):

    def process(self, configfile, infiles, overwrite = False):
        self.loadconfig(configfile)
        for infile in infiles:
            infile = os.path.abspath(infile)
            if not overwrite:
                ename = self.check_skip(infile, self.exportsteps)
                if ename is not None:
                    logger.warning(
                        ("Skipping '{}', rejected by exporter '{}'. Set "
                         "overwrite = True to ignore.").format(infile, ename))
                    continue
            logger.info("Processing '{}'".format(infile))
            try:
                img = self.loadimage(infile)
            except IOError, fire.errors.ImageCannotBeLoadedError:
                logger.error("Unable to load '{}'".format(infile))
                continue
            img = self.preprocess(img, self.preprocsteps)
            # DEBUG
            #import numpy as np
            #np.save("testdata/img_preproc_cropped.npy", img)
            #from fire.debug import showimg, saveimg
            #showimg(img)
            #exit()
            #saveimg("testdata/img_preproc_cropped.png", img)
            detected = self.processchain(img, self.procsteps)
            self.export(infile, detected, self.exportsteps)

    def loadconfig(self, configfile):
        logger.debug("Loading configuration from '{}'".format(configfile))
        # Lazy load yaml so we don't require it
        if configfile.endswith('.yml') or configfile.endswith('.yaml'):
            import yaml
            with open(configfile, 'r') as f:
                cfg = yaml.load(f)
        else:
            # FIXME: Assume JSON?
            logger.critical("Unknown config file type")
            raise NotImplementedError
        # Set default modules if none given:
        if not 'imgloaders' in cfg:
            cfg['imgloaders'] = [
                'fire.imageloaders.MahotasImageLoader',
                ]
        if not 'preprocessors' in cfg:
            cfg['preprocessors'] = {
                'crop': 'fire.preprocessors.CropPreprocessor',
                'makefloat': 'fire.preprocessors.MakeFloatPreprocessor',
                'greyscale': 'fire.preprocessors.GreyscalePreprocessor',
                'removebg': 'fire.preprocessors.RemoveBackgroundPreprocessor',
                }
        if not 'processors' in cfg:
            cfg['processors'] = {
                'threshold': 'fire.processors.ThresholdProcessor',
                'watershed': 'fire.processors.WatershedProcessor',
                'fcd': 'fire.processors.FCDProcessor',
                'tiledfcd': 'fire.processors.TiledFCDProcessor',
                'diskextend': 'fire.processors.DiskExtendProcessor',
                'removeouter': 'fire.processors.RemoveOuterProcessor',
                'dirtyremoveedges': 'fire.processors.DirtyRemoveEdgesProcessor',
                }
        if not 'exporters' in cfg:
            cfg['exporters'] = {
                'string': 'fire.exporters.StringExporter',
                'save': 'fire.exporters.SaveExporter',
                }
        # Load and set config
        for what in ('imgloaders', ):
            instances = []
            for modclass in cfg[what]:
                modulename, classname = modclass.rsplit('.', 1)
                mod = import_module(modulename)
                instances.append(getattr(mod, classname)(manager = self))
            setattr(self, what, instances)
        for what in ('preprocessors', 'processors', 'exporters'):
            instances = {}
            for modname, modclass in cfg[what].items():
                modulename, objectname = modclass.rsplit('.', 1)
                mod = import_module(modulename)
                obj = getattr(mod, objectname)
                if isinstance(obj, type):
                    # obj is a class and should be initialised
                    instances[modname] = obj(manager = self)
                else:
                    # Assume that obj is a function
                    instances[modname] = obj
            setattr(self, what, instances)
        for what in ('preprocsteps', 'procsteps', 'exportsteps'):
            steplist = []
            for rawstep in cfg[what]:
                if isinstance(rawstep, dict):
                    stepname = rawstep.keys()[0]
                    steplist.append( (stepname, {} if rawstep[stepname] is None
                                                   else rawstep[stepname]) )
                else:
                    steplist.append( (rawstep, {}) )
            setattr(self, what, steplist)

    def check_skip(self, infile, exportsteps):
        for ename, eargs in exportsteps:
            if self.exporters[ename].skip(infile):
                return ename
        return None

    def loadimage(self, infile):
        fileext = os.path.splitext(infile)[1][1:]
        for loader in self.imgloaders:
            if fileext.lower() in loader.supported_file_types:
                return loader.load(infile)
        logger.error(
                "No image loader that can read {} files! (File: {})".format(
                    fileext, infile))
        raise fire.errors.ImageCannotBeLoadedError

    def preprocess(self, img, preprocsteps):
        for ppname, ppargs in preprocsteps:
            logger.debug("Calling preprocessor '{}' with arguments: {}".format(
                         ppname, str(ppargs)))
            img = self.preprocessors[ppname](img, **ppargs)
        return img

    def processchain(self, img, procsteps):
        for p in self.processors.values():
            p.reinit()
        detected = []
        for pname, pargs in procsteps:
            logger.debug("Calling processor '{}' with arguments: {}".format(
                         pname, str(pargs)))
            detected.append(
                self.processors[pname](img, detected, **pargs)
                )
        return detected

    def export(self, infile, detected, exportsteps):
        try:
            os.makedirs(os.path.dirname(fire.exporters.outfilebase(infile)))
        except OSError as e:
            if e.errno != 17:  # directory exists
                raise
        for ename, eargs in exportsteps:
            logger.debug("Calling exporter '{}' with arguments: {}".format(
                         ename, str(eargs)))
            self.exporters[ename](infile, detected, **eargs)

