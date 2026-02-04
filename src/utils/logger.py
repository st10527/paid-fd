"""
Logging utilities for PAID-FD experiments.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Global logger registry
_loggers = {}


def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.
    
    Args:
        name: Logger name (usually method or experiment name)
        log_dir: Directory for log files (default: results/logs)
        level: Logging level
        console: Whether to log to console
        file: Whether to log to file
        
    Returns:
        Configured logger
    """
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    
    # Format
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file:
        if log_dir is None:
            log_dir = "results/logs"
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    _loggers[name] = logger
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a simple one.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]
    return setup_logger(name)


class ProgressLogger:
    """
    Helper for logging training progress.
    
    Usage:
        progress = ProgressLogger(logger, total_rounds=200, log_every=10)
        for round_idx in range(200):
            # ... training ...
            progress.log(round_idx, accuracy=0.45, loss=0.5)
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        total_rounds: int,
        log_every: int = 10
    ):
        self.logger = logger
        self.total_rounds = total_rounds
        self.log_every = log_every
        self.start_time = datetime.now()
    
    def log(self, round_idx: int, **metrics):
        """Log progress if at a logging interval."""
        if round_idx % self.log_every == 0 or round_idx == self.total_rounds - 1:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            
            # Build metric string
            metric_strs = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                          for k, v in metrics.items()]
            
            # Estimate remaining time
            if round_idx > 0:
                time_per_round = elapsed / round_idx
                remaining = time_per_round * (self.total_rounds - round_idx)
                time_str = f"ETA: {remaining/60:.1f}min"
            else:
                time_str = ""
            
            self.logger.info(
                f"Round {round_idx}/{self.total_rounds} | "
                f"{' | '.join(metric_strs)} | "
                f"{time_str}"
            )
    
    def finish(self, **final_metrics):
        """Log final results."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        metric_strs = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                      for k, v in final_metrics.items()]
        
        self.logger.info(
            f"Training completed in {elapsed/60:.1f} minutes | "
            f"{' | '.join(metric_strs)}"
        )
