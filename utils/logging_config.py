import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name, log_dir='logs'):
    """Configure logger with consistent formatting and file/console output"""
    # Create logs directory if it doesn't exist
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler - daily rotating log file
    today = datetime.now().strftime('%Y-%m-%d')
    file_handler = logging.FileHandler(
        log_dir / f'{name}_{today}.log'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger 