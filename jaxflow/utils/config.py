import os
from typing import Any, Dict, Optional

class Config:
    """
    Simple configuration management using environment variables and defaults.
    """
    
    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """Get a string value from environment variables."""
        return os.environ.get(key, default)
    
    @staticmethod
    def get_int(key: str, default: int = 0) -> int:
        """Get an integer value from environment variables."""
        try:
            val = os.environ.get(key)
            if val is None:
                return default
            return int(val)
        except (ValueError, TypeError):
            return default
            
    @staticmethod
    def get_bool(key: str, default: bool = False) -> bool:
        """Get a boolean value from environment variables."""
        val = os.environ.get(key)
        if val is None:
            return default
        return val.lower() in ("true", "1", "yes", "on", "t")

# Common Configuration Constants
JAXFLOW_DATA_DIR = Config.get("JAXFLOW_DATA_DIR", "./data")
JAXFLOW_CACHE_DIR = Config.get("JAXFLOW_CACHE_DIR", "./.cache")
JAXFLOW_NUM_WORKERS = Config.get_int("JAXFLOW_NUM_WORKERS", 4)
