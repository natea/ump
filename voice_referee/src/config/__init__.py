"""Configuration module for Voice Referee."""

from .settings import (
    DailyConfig,
    DeepgramConfig,
    LLMConfig,
    TTSConfig,
    ProcessorConfig,
    Settings,
)
from .daily_config import DailyConfig as DailyConfigAlt

__all__ = [
    "DailyConfig",
    "DeepgramConfig",
    "LLMConfig",
    "TTSConfig",
    "ProcessorConfig",
    "Settings",
]
