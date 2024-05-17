"""
@file src/utils/logger.py
@brief This module contains functions for setting up the logger.

Functions:
    - setupCustomLogger(log_level: str) -> None:
        Sets up the logger with the specified log level. This function sets up the logger with the specified log level and formats the log messages.

Note:
    This module is designed to be used in the main script to set up the logger for logging messages.
"""

import logging
import os
from sys import stdout


def setupCustomLogger(log_level):
    """
    @brief Sets up the logger with the specified log level.

    This function sets up the logger with the specified log level and formats the log messages.

    @param log_level The log level to be set for the logger.
    @return None
    """
    LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    logging.basicConfig(
    level=LEVELS.get(log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(os.environ['PROJECT_LOG_DIR'], 'log.log')),
        logging.StreamHandler(stdout)
    ]
)