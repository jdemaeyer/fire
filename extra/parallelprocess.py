import os
import sys
import multiprocessing

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from OIP.manager import ProcessingManager

from OIP.logging import logger
from logging import INFO
logger.setLevel(INFO)


# CONFIG

#BASEDIR = "/run/media/jakob/TOSHIBA EXT/Data/Small/20150210_04-08/"
#PROCESSCONFS = [
#    # ( Folder, First File )
#    ( "04", "DSC_0016.JPG" ),
#    ( "05", "DSC_0197.JPG" ),
#    ( "07", "DSC_0559.JPG" ),
#    ( "08", "DSC_0741.JPG" ),
#]

POOLSIZE = 3
LISTPARTS = 40
BASEDIR = "/run/media/jakob/TOSHIBA EXT/Data/Big/"
PROCESSCONFS = [
    # ( Image Folder, First File, Last File, Config File )
    ( "20150218/Full/100NCD90/", "DSC_0020.JPG", "DSC_0771.JPG", "20150218/Full/imageproc.yml" ),
    #( "20150218/Full/100NCD90/", "DSC_0020.JPG", "DSC_0771.JPG", "20150218/Full/imageproc.yml" ),
    #( "20150218/Full/100NCD90/", "DSC_0020.JPG", "DSC_0771.JPG", "20150218/Full/imageproc.yml" ),
    #( "20150218/Full/100NCD90/", "DSC_0020.JPG", "DSC_0771.JPG", "20150218/Full/imageproc.yml" ),
    #( "20150220/Full/100NCD90/", "DSC_0010.JPG", "DSC_0814.JPG", "20150220/Full/imageproc.yml" ),
    ( "20150220/Full/100NCD90/", "DSC_0010.JPG", "DSC_0814.JPG", "20150220/Full/imageproc.yml" ),
    ( "20150225/Full/100NCD90/", "DSC_0016.JPG", "DSC_0775.JPG", "20150225/Full/imageproc.yml" ),
    ( "20150227/Full/100NCD90/", "DSC_0011.JPG", "DSC_0801.JPG", "20150227/Full/imageproc.yml" ),
    ( "20150228/Full/100NCD90/", "DSC_0013.JPG", "DSC_0792.JPG", "20150228/Full/imageproc.yml" ),
]
OVERWRITE = False


# BUILD QUEUES

queues = []
for proc_index, (folder, firstfile, lastfile, configfile) in enumerate(PROCESSCONFS):
    imgpath = os.path.join(BASEDIR, folder)
    filelist = [ f for f in os.listdir(imgpath)
                   if os.path.isfile(os.path.join(imgpath, f)) ]
    filelist = sorted(filelist)
    filelist = filelist[filelist.index(firstfile):filelist.index(lastfile)]
    shortlist = []
    #for i in (0, 10, 15, 20, 40, 60, 80, 100, 130, 160):
    #for i in range(75, 50, -5):
    #for i in range(0, len(filelist), 16):
    #    shortlist.append(filelist[i])
    fileindices = []
    fileindices.extend( range(0, 300, 10) )
    fileindices.extend( range(300, 500, 5) )
    #fileindices.extend( range(500, 750, 5) )
    fileindices.extend( range(500, len(filelist), 2) )
    #for i in fileindices[proc_index%2::2]:
    for i in fileindices:
        shortlist.append(filelist[i])
    print len(shortlist)
    fullpathlist = [ os.path.join(imgpath, f) for f in shortlist ]
    configfullpath = os.path.join(BASEDIR, configfile)
    for i in range(LISTPARTS):
        partlen = float(len(fullpathlist)) / LISTPARTS
        start = int(round(i * partlen))
        end = int(round((i+1) * partlen))
        queues.append((configfullpath, fullpathlist[start:end]))


print "{} queues containing {} images".format(
        len(queues),
        sum( ( len(q[1]) for q in queues ) )
        )

import time
time.sleep(2)

# THREADING SETUP

def processqueue(queue):
    pm = ProcessingManager()
    pm.process(queue[0], queue[1], overwrite = OVERWRITE)


# PROCESS ALL QUEUES WITH WORKER POOL

pool = multiprocessing.Pool(processes = POOLSIZE)
pool.map(processqueue, queues)

