import logging
from colorlog import ColoredFormatter
import os
from datetime import datetime


def logger_setup(name="trade_logger", level=logging.INFO, log_to_file=True,
                 code=None, year=None, flow=None, log_dir=None):
    """
    Set up a color console logger with optional file output.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # === Always attach console stream handler (only once)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler()
        formatter = ColoredFormatter(
            "%(log_color)s[%(asctime)s] %(levelname)s%(reset)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG":    "cyan",
                "INFO":     "green",
                "WARNING":  "yellow",
                "ERROR":    "red",
                "CRITICAL": "bold_red",
            }
        )
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # === Force-add file handler if all info is provided
    if log_to_file and code and year and flow and log_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"hs_{code}_{year}_{flow}_{timestamp}.log"
        log_path = os.path.join(log_dir, log_filename)

        # Avoid duplicate file handler
        if not any(isinstance(h, logging.FileHandler) and getattr(h, '_log_path', None) == log_path for h in logger.handlers):
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
            file_handler._log_path = log_path  # for tracking uniqueness
            logger.addHandler(file_handler)
            logger._log_path = log_path  # you can access this later
            print(f"[INFO] Log file created: {log_path}")

    return logger