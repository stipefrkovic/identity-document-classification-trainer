import logging
import os

# Retrieve log level from environment variable.
log_level = os.environ['LOG_LEVEL'].upper()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add a FileHandler to the logger.
log_file = '/app/logfiles/log.txt'
f_handler = logging.FileHandler(log_file)
f_handler.setLevel(log_level)
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
f_handler.setFormatter(f_format)
logger.addHandler(f_handler)

c_handler = logging.StreamHandler()
c_handler.setLevel(log_level)
c_format = logging.Formatter('%(levelname)s - %(message)s')

c_handler.setFormatter(c_format)
logger.addHandler(c_handler)