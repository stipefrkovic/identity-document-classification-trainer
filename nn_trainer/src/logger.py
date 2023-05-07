import logging
import os

# Retrieve log level from environment variable.
log_level = os.environ['LOG_LEVEL'].upper()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


c_handler = logging.StreamHandler()
c_handler.setLevel(log_level)
c_format = logging.Formatter('%(levelname)s - %(message)s')

c_handler.setFormatter(c_format)
logger.addHandler(c_handler)