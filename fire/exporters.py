"""
FIRE components that export DetectionInfos (to files, databases, ...)
"""

import os.path
import numpy as np
from FIRE.base import Exporter


def outfilebase(infile):
    inbase, intail = os.path.split(infile)
    outfilebase = os.path.join(inbase, "processed/", intail)
    return outfilebase


class StringExporter(Exporter):

    def __call__(self, infile, detectioninfos, **kwargs):
        outfile = outfilebase(infile) + ".txt"
        with open(outfile, 'w') as f:
            for procdetected in detectioninfos:
                for d in procdetected:
                    f.write(str(d))
                    f.write("\n")

    def skip(self, infile):
        return os.path.isfile(outfilebase(infile) + ".txt")


class SaveExporter(Exporter):

    def __call__(self, infile, detectioninfos, **kwargs):
        saveindex = 0
        ofb = outfilebase(infile)
        for procdetected in detectioninfos:
            for d in procdetected:
                if hasattr(d, 'save'):
                    d.save(ofb + "_save_{}".format(saveindex))
                    saveindex += 1
