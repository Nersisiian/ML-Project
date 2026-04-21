import logging
import structlog
from structlog.processors import JSONRenderer, TimeStamper
from typing import Dict, Any
import sys

def setup_logging(debug: bool = False):
    """Configure structured logging"""
    
    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if debug:
        # Console logging for development
        structlog.configure(
            processors=shared_processors + [
                structlog.dev.ConsoleRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        logging.basicConfig(
            format="%(message)s",
            level=logging.DEBUG,
            stream=sys.stdout
        )
    else:
        # JSON logging for production
        structlog.configure(
            processors=shared_processors + [
                JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        logging.basicConfig(
            format="%(message)s",
            level=logging.INFO,
            stream=sys.stdout
        )

def get_logger(name: str):
    """Get structured logger"""
    return structlog.get_logger(name)