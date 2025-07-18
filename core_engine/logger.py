"""
🌌 INFINITUS Logging System
Advanced logging system for the ultimate sandbox survival crafting game.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional
from datetime import datetime
# Simplified color support
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    RESET = '\033[0m'

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    
    # Color mappings for different log levels
    COLORS = {
        'DEBUG': Colors.CYAN,
        'INFO': Colors.GREEN,
        'WARNING': Colors.YELLOW,
        'ERROR': Colors.RED,
        'CRITICAL': Colors.RED,
    }
    
    def format(self, record):
        # Get the color for this log level
        color = self.COLORS.get(record.levelname, '')
        
        # Format the message
        formatted = super().format(record)
        
        # Add color if we have one
        if color:
            formatted = f"{color}{formatted}{Colors.RESET}"
        
        return formatted

class GameLogger:
    """Advanced logging system for the game engine."""
    
    def __init__(self, name: str = "infinitus", log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for log files
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup loggers
        self.main_logger = self._setup_main_logger()
        self.performance_logger = self._setup_performance_logger()
        self.error_logger = self._setup_error_logger()
        self.debug_logger = self._setup_debug_logger()
    
    def _setup_main_logger(self) -> logging.Logger:
        """Setup the main game logger."""
        logger = logging.getLogger(f"{self.name}.main")
        logger.setLevel(logging.INFO)
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler
        file_handler = logging.FileHandler(
            self.log_dir / f"infinitus_{self.timestamp}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def _setup_performance_logger(self) -> logging.Logger:
        """Setup performance monitoring logger."""
        logger = logging.getLogger(f"{self.name}.performance")
        logger.setLevel(logging.INFO)
        
        # Performance log file
        handler = logging.FileHandler(
            self.log_dir / f"performance_{self.timestamp}.log",
            encoding='utf-8'
        )
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s | %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _setup_error_logger(self) -> logging.Logger:
        """Setup error logging."""
        logger = logging.getLogger(f"{self.name}.error")
        logger.setLevel(logging.ERROR)
        
        # Error log file
        handler = logging.FileHandler(
            self.log_dir / f"errors_{self.timestamp}.log",
            encoding='utf-8'
        )
        handler.setLevel(logging.ERROR)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s\n'
            'Exception: %(exc_info)s\n'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _setup_debug_logger(self) -> logging.Logger:
        """Setup debug logging."""
        logger = logging.getLogger(f"{self.name}.debug")
        logger.setLevel(logging.DEBUG)
        
        # Debug log file
        handler = logging.FileHandler(
            self.log_dir / f"debug_{self.timestamp}.log",
            encoding='utf-8'
        )
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def info(self, message: str, category: str = "main"):
        """Log an info message."""
        if category == "performance":
            self.performance_logger.info(message)
        else:
            self.main_logger.info(message)
    
    def warning(self, message: str):
        """Log a warning message."""
        self.main_logger.warning(message)
    
    def error(self, message: str, exc_info=None):
        """Log an error message."""
        self.main_logger.error(message, exc_info=exc_info)
        self.error_logger.error(message, exc_info=exc_info)
    
    def debug(self, message: str):
        """Log a debug message."""
        self.main_logger.debug(message)
        self.debug_logger.debug(message)
    
    def critical(self, message: str, exc_info=None):
        """Log a critical message."""
        self.main_logger.critical(message, exc_info=exc_info)
        self.error_logger.critical(message, exc_info=exc_info)
    
    def performance(self, message: str):
        """Log a performance message."""
        self.performance_logger.info(message)
    
    def set_level(self, level: str):
        """Set the logging level."""
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        
        if level.upper() in level_map:
            self.main_logger.setLevel(level_map[level.upper()])
            self.debug_logger.setLevel(level_map[level.upper()])

# Global logger instance
_game_logger: Optional[GameLogger] = None

def setup_logging(name: str = "infinitus", log_dir: str = "logs", level: str = "INFO") -> GameLogger:
    """Setup the global game logging system."""
    global _game_logger
    
    if _game_logger is None:
        _game_logger = GameLogger(name, log_dir)
        _game_logger.set_level(level)
        
        # Log startup message
        _game_logger.info("🌌 INFINITUS Logging System Initialized")
        _game_logger.info(f"Log directory: {Path(log_dir).absolute()}")
        _game_logger.info(f"Log level: {level}")
    
    return _game_logger

def get_logger() -> Optional[GameLogger]:
    """Get the global game logger."""
    return _game_logger

def log_system_info():
    """Log system information."""
    if _game_logger:
        import platform
        import os
        
        _game_logger.info("=== SYSTEM INFORMATION ===")
        _game_logger.info(f"OS: {platform.system()} {platform.release()}")
        _game_logger.info(f"Python: {platform.python_version()}")
        _game_logger.info(f"Architecture: {platform.machine()}")
        _game_logger.info(f"Processor: {platform.processor()}")
        _game_logger.info(f"CPU Cores: {os.cpu_count()}")
        _game_logger.info("=== END SYSTEM INFO ===")

def log_performance(fps: float, frame_time: float, memory_usage: float):
    """Log performance metrics."""
    if _game_logger:
        _game_logger.performance(f"FPS: {fps:.1f} | Frame Time: {frame_time:.3f}ms | Memory: {memory_usage:.1f}MB")

def log_error_with_context(error: Exception, context: str = ""):
    """Log an error with additional context."""
    if _game_logger:
        message = f"Error in {context}: {str(error)}" if context else str(error)
        _game_logger.error(message, exc_info=True)

def log_startup_banner():
    """Log the game startup banner."""
    if _game_logger:
        banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║    🌌 INFINITUS: The Ultimate Sandbox Survival Crafting Game    ║
║                                                                  ║
║    Starting up the most powerful open-world game engine...      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
        """
        for line in banner.strip().split('\n'):
            _game_logger.info(line)

def cleanup_logs(days_to_keep: int = 7):
    """Clean up old log files."""
    if _game_logger:
        import time
        
        log_dir = _game_logger.log_dir
        current_time = time.time()
        cutoff_time = current_time - (days_to_keep * 24 * 60 * 60)
        
        for log_file in log_dir.glob("*.log"):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    _game_logger.info(f"Cleaned up old log file: {log_file.name}")
                except Exception as e:
                    _game_logger.warning(f"Failed to delete old log file {log_file.name}: {e}")