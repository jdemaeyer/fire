from __future__ import absolute_import

import logging


# Basic setup
logging.basicConfig(
            format = '%(asctime)s  %(levelname)s  %(message)s',
            datefmt = '%Y-%m-%d %H:%M:%S',
            level = logging.INFO
            )

# Colour logging
# Hacked together quick n dirty from https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
# TODO: It's probably gonna write the color sequences to files also?
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"
colorseq = lambda mycolor: COLOR_SEQ % (30 + mycolor)
logging.addLevelName( logging.DEBUG, colorseq(BLUE) + logging.getLevelName(logging.DEBUG) + RESET_SEQ)
logging.addLevelName( logging.INFO, colorseq(WHITE) + logging.getLevelName(logging.INFO) + RESET_SEQ)
logging.addLevelName( logging.WARNING, colorseq(YELLOW) + logging.getLevelName(logging.WARNING) + RESET_SEQ)
logging.addLevelName( logging.ERROR, colorseq(RED) + logging.getLevelName(logging.ERROR) + RESET_SEQ)
logging.addLevelName( logging.CRITICAL, BOLD_SEQ + colorseq(RED) + logging.getLevelName(logging.CRITICAL) + RESET_SEQ)

# My logger
logger = logging.getLogger("fire")

logging.captureWarnings(True)
# TODO: Get warnings from py.warnings logger and set their level to DEBUG

