import os

import numpy as np

from fire.manager import ProcessingManager
from fire.exporters import outfilebase
from fire.detectioninfos import (DiskInfo, FreeAreaInfo,
                                NonOverlappingFreeAreasInfo)


def get_img_list(imgfolder = None, procfolder = None):
    """Return list of _processed_ images"""
    if procfolder is None and imgfolder is None:
        raise ValueError("Need to provide either image or processed folder")
    elif procfolder is None:
        procfolder = os.path.join(imgfolder, "processed/")
    elif imgfolder is None:
        imgfolder = os.path.dirname(procfolder)
    return sorted([ os.path.join(imgfolder, f[:-4])
          for f in os.listdir(procfolder)
          if os.path.isfile(os.path.join(procfolder, f))
          and f.endswith(".txt") ])

def get_procfiles(imgfile, procfolder = None):
    if procfolder is None:
        procfilebase = outfilebase(imgfile)
        procfolder = os.path.dirname(procfilebase)
    else:
        procfilebase = os.path.join(procfolder, os.path.basename(imgfile))
    procfile_candidates = [ os.path.join(procfolder, f)
                            for f in os.listdir(procfolder) ]
    procfiles = [ pc for pc in procfile_candidates
                  if os.path.isfile(pc)
                  and pc.startswith(procfilebase) ]
    return procfiles

def read_all_nofais(imgfile, procfolder = None):
    procfiles = [ x for x in get_procfiles(imgfile, procfolder)
                  if "_save_" in os.path.basename(x) ]
    allnofais = None
    for procfile in procfiles:
        if not procfile.endswith('.npz'):
            continue
        with np.load(procfile) as f:
            try:
                areas = f['nofai']
            except KeyError:
                # Unable to extract NOFAI, probably not of type NOFAI
                continue
        if allnofais is None:
            allnofais = areas.astype(np.uint16)
        else:
            oldmax = allnofais.max()
            where = np.where(areas != 0)
            allnofais[where] = areas[where]
            allnofais[where] += oldmax
    return allnofais

def file_lines_starting_with(filename, startstring = ''):
    """Generator that reads lines from a file and yields only these starting
    with a given string"""
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith(startstring):
                yield line

def get_all_detections(imgfile, procfolder = None, load_txts = True,
        load_npzs = True):
    procfiles = get_procfiles(imgfile, procfolder)
    all_detections = []
    for procfile in procfiles:
        if procfile.endswith('.txt') and load_txts:
            circles = np.loadtxt(file_lines_starting_with(procfile, 'd'),
                                     usecols = (2, 3, 4))
            for c in circles:
                all_detections.append(DiskInfo(c[1], c[0], c[2]))
        elif procfile.endswith('.npz') and load_npzs:
            for detinfo in (FreeAreaInfo, NonOverlappingFreeAreasInfo):
                try:
                    all_detections.append(detinfo.load(procfile))
                except KeyError:
                    pass
    return all_detections

def get_preprocessed_img(imgfile, configfile):
    pm = ProcessingManager()
    pm.loadconfig(configfile)
    img = pm.loadimage(imgfile)
    return pm.preprocess(img, pm.preprocsteps)

def rand_labels(labeled):
    import mahotas as mh
    relabeled, _ = mh.labeled.relabel(labeled)
    maxlabel = relabeled.max()
    newlabels = np.random.permutation(np.arange(1, maxlabel+1))
    labeled_rand = np.zeros_like(relabeled)
    for i in range(1, maxlabel+1):
        labeled_rand[relabeled == i] = newlabels[i-1]
    return labeled_rand

