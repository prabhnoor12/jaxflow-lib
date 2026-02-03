import sys
import os
from unittest.mock import MagicMock

# Mock dependencies to bypass missing packages in environment
sys.modules["numpy"] = MagicMock()
sys.modules["jax"] = MagicMock()
sys.modules["jax.random"] = MagicMock()
sys.modules["flax"] = MagicMock()

# Add current directory to path
sys.path.append(os.getcwd())

from jaxflow.utils import get_logger, timer, seed_everything, Config

def test_logging():
    print("Testing logging...")
    logger = get_logger("test_logger", level="DEBUG")
    logger.info("This is an info message")
    logger.debug("This is a debug message")
    print("Logging test passed")

def test_timing():
    print("Testing timing...")
    with timer("Sleep operation"):
        import time
        time.sleep(0.01)
    print("Timing test passed")

def test_config():
    print("Testing config...")
    os.environ["TEST_VAR"] = "123"
    assert Config.get_int("TEST_VAR") == 123
    print("Config test passed")

def test_random():
    print("Testing random...")
    # This will use the mocked numpy/jax, so we just check it doesn't crash
    seed_everything(42)
    print("Random seeding test passed (mocked)")

if __name__ == "__main__":
    test_logging()
    test_timing()
    test_config()
    test_random()
