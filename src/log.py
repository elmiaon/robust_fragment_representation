import logging
import os
import sys
import pytz
from datetime import datetime, timezone
FORMATTER = logging.Formatter("[%(filename)s:%(lineno)4s - %(funcName)s()] — %(levelname)s — %(message)s")
# tz = pytz.timezone("Asia/Bangkok")
# month, date, dtime = f"{datetime.now(tz).strftime('%b %d %H:%M:%S')}".split(" ")
# utc_dt = datetime.now(timezone.utc)
month, date, dtime = f"{datetime.now().strftime('%b %d %H:%M:%S')}".split(" ")
# if not os.path.exists(f"logs/{month}/"):
#    os.mkdir(f"logs/{month}/")
# if not os.path.exists(f"logs/{month}/{date}/"):
#    os.mkdir(f"logs/{month}/{date}/")
LOG_LEVEL = None
FILENAME = None

def init_logger(file, name, log_level):
   global FILENAME, LOG_LEVEL
   FILENAME = file[:-3]
   LOG_LEVEL = log_level
   logger = get_logger(name)
   return logger

def get_console_handler():
   console_handler = logging.StreamHandler(sys.stdout)
   console_handler.setFormatter(FORMATTER)
   return console_handler
   
def get_file_handler():
   if not os.path.exists("logs/"):
      os.mkdir("logs/")
   if not os.path.exists(f"logs/{FILENAME}/"):
      os.mkdir(f"logs/{FILENAME}")
   LOG_FILE = f"logs/{FILENAME}/{month},{date}-{dtime}.{FILENAME}.log"
   file_handler = logging.FileHandler(LOG_FILE)
   file_handler.setFormatter(FORMATTER)
   return file_handler

def get_logger(logger_name):
   logger = logging.getLogger(logger_name)
   if logger.hasHandlers():
      return logger
   else:
      logger.setLevel(LOG_LEVEL) # better to have too much log than not enough
      logger.addHandler(get_console_handler())
      logger.addHandler(get_file_handler())
      # with this pattern, it's rarely necessary to propagate the error up to parent
      logger.propagate = False
      return logger

