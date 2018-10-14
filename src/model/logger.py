import sys
import logging


class Logger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        format = logging.Formatter("%(asctime)s - %(message)s")
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setFormatter(format)
        self.logger.addHandler(sh)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)