import logging
import sys
import os
import platform
import torch
import transformers
from typing import Optional, Dict, Any

def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    rank: int = 0
) -> logging.Logger:
    """
    Setup logging configuration for the project.
    
    Args:
        log_level: Logging level (default: logging.INFO)
        log_file: Optional path to a log file
        rank: Process rank (default: 0). Only rank 0 will log to console/file by default.
        
    Returns:
        Configured root logger
    """
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Only add handlers if rank is 0 (master process)
    if rank == 0:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(console_handler)

        # File handler (if requested)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(file_handler)
    else:
        # For non-master processes, maybe just a NullHandler or a restricted handler
        # Usually we want to silence them to avoid interleaved logs
        root_logger.addHandler(logging.NullHandler())

    # Reduce verbosity of some third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    return root_logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)

def log_system_info(logger: logging.Logger):
    """Log relevant system information for reproducibility."""
    logger.info("=" * 40)
    logger.info("System Information")
    logger.info("=" * 40)
    logger.info(f"OS: {platform.system()} {platform.release()}")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"Transformers Version: {transformers.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA Available: Yes")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("CUDA Available: No")
    logger.info("-" * 40)

def log_model_info(logger: logging.Logger, model_config: Dict[str, Any]):
    """Log model architecture and configuration details."""
    logger.info("=" * 40)
    logger.info("Model Configuration")
    logger.info("=" * 40)
    for key, value in model_config.items():
        logger.info(f"{key}: {value}")
    logger.info("-" * 40)

def log_training_setup(logger: logging.Logger, training_config: Dict[str, Any]):
    """Log training hyperparameters."""
    logger.info("=" * 40)
    logger.info("Training Setup")
    logger.info("=" * 40)
    for key, value in training_config.items():
        logger.info(f"{key}: {value}")
    logger.info("-" * 40)