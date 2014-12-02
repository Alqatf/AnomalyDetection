# -*- coding: utf-8 -*-
import logging
import util
""" Level
DEBUG   Detailed information, typically of interest only when diagnosing problems.
INFO    Confirmation that things are working as expected.
WARNING An indication that something unexpected happened, or indicative of some problem in the near future (e.g. ‘disk space low’). The software is still working as expected.
ERROR   Due to a more serious problem, the software has not been able to perform some function.
CRITICAL    A serious error, indicating that the program itself may be unable to continue running.
https://docs.python.org/2/howto/logging.html
"""

def set_file(path):
    util.delete_file(path)
    logging.basicConfig(format='%(levelname)s:%(message)s',filename=path,level=logging.DEBUG)

def debug(txt):
    logging.debug(txt)

def info(txt):
    logging.info(txt)

def warning(txt):
    logging.warning(txt)

if __name__ == '__main__':
    path = util.make_today_folder('./results')
    set_file(path + "/log.txt")
    logging.debug("asfdsf")
    logging.info("asfdsf")
    logging.warning("asfdsf")
