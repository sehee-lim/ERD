from pathlib import Path
import logging
import os


class Logger:
    # does not return anything
    def __init__(self, _name, _root_dir, level = logging.DEBUG, ) -> None:
				# convert _root_dir string to a Path object
        self._root_dir = Path(_root_dir)
        os.makedirs(self._root_dir, exist_ok = True)
        
				# create a logger object for logging
        self.logger = logging.getLogger(_name)
        
				# set the default level of the logger
        self.logger.setLevel(level)


		# Handler: write log messages to a file
    def addFileHandler(self,
                       fname: str,
                       level = None,
                       _format = "[%(asctime)s, %(levelname)s] : %(message)s"):
				# create a Handler
        handler = logging.FileHandler(self._root_dir / fname, encoding = 'utf-8')
				
				# specify the format of log messages
        formatter = logging.Formatter(_format)

        if level is not None:
            handler.setLevel(level)
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)


    def debug(self, line):
				# line: message to be recoreded in the log
        self.logger.debug(line)   # log the message at DEBUG level


    def info(self, line):
        self.logger.info(line)   # log the message at INFO level
        
    def error(self, line):
      self.logger.error(line)   # log the message at ERROR level
        