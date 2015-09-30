FIRE
====

FIRE is a framework that eases the organisation of batch image processing.
Images are run through a chain of loaders, preprocessors, processors, and
exporters.


Usage
-----

Example usage can be found in extra/singleprocess.py and
extra/parallelprocess.py, with an example configuration file in
extra/example_config.yml.


Dependencies
------------

*   Mandatory:
    *   python2
    *   numpy
*   Strongly recommended:
    *   mahotas -- most of the standard preprocessors and image loaders use this
        sweet little library ([PyPI](https://pypi.python.org/pypi/mahotas),
        [Homepage](http://luispedro.org/software/mahotas/))
*   Optional
    *   matplotlib -- for showing images, useful for debugging
    *   pillow/PIL -- for saving images, useful for debugging
