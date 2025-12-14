"""Custom frames for Voice Referee vision pipeline."""

from src.frames.vision_frames import (
    ScreenAnalysisFrame,
    CommentaryRequestFrame,
    VisionErrorFrame,
)

__all__ = [
    "ScreenAnalysisFrame",
    "CommentaryRequestFrame",
    "VisionErrorFrame",
]
