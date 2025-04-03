# backend/utils/logging_config.py (Example using basicConfig)

import logging
import sys
import os # Keep os import if needed elsewhere, but remove LOG_DIR specifics

# Removed: LOG_DIR = "logs"
# Removed: os.makedirs(LOG_DIR, exist_ok=True)

def setup_logging(log_level=logging.INFO):
    """Sets up basic logging to stdout."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        # Use stream=sys.stdout or remove handler config to use default stderr
        stream=sys.stdout,
        # Removed: handlers=[ ... FileHandler ... ]
    )
    # Optionally silence noisy libraries
    # logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

def get_logger(name: str):
    """Gets a logger instance."""
    return logging.getLogger(name)

# Call setup when the module is imported (or call it explicitly in main.py)
# setup_logging() # Or call setup_logging() once in backend/main.py