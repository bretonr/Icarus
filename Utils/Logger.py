# Licensed under a 3-clause BSD style license - see LICENSE

#import os
import logging

## Possible levels are 'INFO', 'DEBUG', a numeral between 5-9.
LEVEL = 'INFO'
#LEVEL = 'DEBUG'

for i in xrange(5,9):
    logging.addLevelName(i, "DEBUG{}".format(i))

if False:
    ## Configurating the logger for a file and console output
    config_folder = os.path.join(os.environ['HOME'], '.icarus')
    logging.basicConfig(format='%(asctime)s | %(name)-30s | %(funcName)-25s | %(levelname)-6s | %(message)s', datefmt='%m-%d %H:%M', filename=os.path.join(archive_folder, 'log.txt'), filemode='a')
    ## Configurating the logger for the console
    ## define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)-30s %(funcName)-25s: %(levelname)-6s %(message)s')
    ## tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
else:
    ## Configurating the logger for console output only
    logging.basicConfig(level=LEVEL, format='%(name)-30s %(funcName)-25s: %(levelname)-6s %(message)s', datefmt='%m-%d %H:%M')



