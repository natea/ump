"""
Pytest configuration and shared fixtures for voice_referee tests
"""

import pytest
import sys
from pathlib import Path

# Add voice_referee's parent directory (ump) to Python path for imports
# This allows imports like: from voice_referee.src.analysis import ...
voice_referee_parent = Path(__file__).parent.parent.parent
sys.path.insert(0, str(voice_referee_parent))

# Also add src directory for direct imports like: from analysis import ...
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture providing path to test data directory"""
    return Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration between tests"""
    import logging
    # Clear all handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Reset log level
    logging.root.setLevel(logging.WARNING)
    yield
