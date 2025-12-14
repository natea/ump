"""
Custom frame types for the vision pipeline.

These frames are used for communication between vision processors
and provide type-safe data structures for screen analysis.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any

from pipecat.frames.frames import Frame


@dataclass
class ScreenAnalysisFrame(Frame):
    """Frame containing screen analysis results from vision AI.

    This frame is produced by ScreenAnalysisProcessor after analyzing
    a screen capture and contains the AI's description and metadata.

    Attributes:
        description: Natural language description of screen content
        confidence: Model's confidence in analysis (0.0-1.0)
        timestamp: Unix timestamp when analysis was performed
        frame_number: Sequential frame number from capture
        image_size: Tuple of (width, height) of analyzed image
        provider: Vision provider used (anthropic, openai, google)
        model: Model identifier used
        latency_ms: Time taken for analysis in milliseconds
        tokens_used: Number of tokens consumed
        cost_usd: Estimated cost of this analysis
        requires_commentary: Whether this analysis should trigger commentary
        detected_elements: List of detected screen elements
    """
    description: str
    confidence: float = 0.0
    timestamp: float = 0.0
    frame_number: int = 0
    image_size: tuple = (0, 0)
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0
    requires_commentary: bool = False
    detected_elements: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"ScreenAnalysisFrame("
            f"desc={self.description[:50]}..., "
            f"conf={self.confidence:.2f}, "
            f"frame={self.frame_number})"
        )


@dataclass
class CommentaryRequestFrame(Frame):
    """Frame requesting commentary generation.

    This frame is used when the system wants to trigger
    spoken commentary, carrying context for the LLM.

    Attributes:
        analysis: Source screen analysis
        context: Additional context for commentary
        priority: Priority level (higher = more urgent)
        style: Commentary style override (concise, detailed, technical)
    """
    analysis: Optional[ScreenAnalysisFrame] = None
    context: str = ""
    priority: int = 0
    style: str = "concise"

    def __str__(self) -> str:
        return (
            f"CommentaryRequestFrame("
            f"priority={self.priority}, "
            f"style={self.style})"
        )


@dataclass
class VisionErrorFrame(Frame):
    """Frame indicating a vision processing error.

    This frame is emitted when vision processing fails,
    allowing downstream processors to handle gracefully.

    Attributes:
        error_type: Type of error (api_error, timeout, cost_exceeded, etc.)
        error_message: Human-readable error description
        timestamp: Unix timestamp when error occurred
        frame_number: Frame that caused the error
        recoverable: Whether processing can continue
    """
    error_type: str
    error_message: str
    timestamp: float = 0.0
    frame_number: int = 0
    recoverable: bool = True

    def __str__(self) -> str:
        return (
            f"VisionErrorFrame("
            f"type={self.error_type}, "
            f"recoverable={self.recoverable})"
        )


@dataclass
class VisionMetricsFrame(Frame):
    """Frame containing vision pipeline metrics.

    Periodically emitted to track vision pipeline performance.

    Attributes:
        frames_processed: Total frames received
        analyses_performed: Total analyses completed
        total_cost_usd: Total cost so far
        average_latency_ms: Average analysis latency
        error_count: Number of errors encountered
        timestamp: Metrics collection timestamp
    """
    frames_processed: int = 0
    analyses_performed: int = 0
    total_cost_usd: float = 0.0
    average_latency_ms: float = 0.0
    error_count: int = 0
    timestamp: float = 0.0

    def __str__(self) -> str:
        return (
            f"VisionMetricsFrame("
            f"analyses={self.analyses_performed}, "
            f"cost=${self.total_cost_usd:.4f})"
        )
