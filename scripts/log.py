#!/usr/bin/env python3

import loguru, sys
from loguru import logger

logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>", backtrace=True, level="INFO")
logger.add("file.log", rotation="12:00")
logger.debug('yes')

@logger.catch
def func(a, b):
    return a / b

func(0,0)
