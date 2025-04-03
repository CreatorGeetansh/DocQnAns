import logging
import sys
from logging.handlers import TimedRotatingFileHandler
import os

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# --- Configuration ---
LOG_LEVEL = logging.INFO  # Change to logging.DEBUG for more verbose output
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# --- Create Formatter ---
formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

# --- Create Handlers ---
# Console Handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

# File Handler (Rotates logs daily, keeps 7 days of backup)
file_handler = TimedRotatingFileHandler(
    LOG_FILE, when="midnight", interval=1, backupCount=7, encoding="utf-8"
)
file_handler.setFormatter(formatter)

# --- Configure Root Logger ---
def setup_logging():
    """Configures the root logger."""
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)

    # Remove existing handlers if any (to avoid duplicates in frameworks like Uvicorn)
    # for handler in root_logger.handlers[:]:
    #     root_logger.removeHandler(handler)

    if not root_logger.handlers: # Add handlers only if they haven't been added
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        root_logger.info("Logging configured.")

# --- Get Logger ---
def get_logger(name: str) -> logging.Logger:
    """Gets a logger instance with the specified name."""
    return logging.getLogger(name)