"""
Screen Analysis Processor - Analyzes screen content for voice commentary.

This processor receives UserImageRawFrame objects from Daily transport's
screen capture, analyzes the content using vision AI, and emits
ScreenAnalysisFrame for the CommentaryProcessor to handle.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

from pipecat.frames.frames import Frame, UserImageRawFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from src.services.vision_service import VisionService, VisionAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class ScreenAnalysisFrame(Frame):
    """Frame containing screen analysis results.

    This frame is emitted when screen content has been analyzed
    and can be used by downstream processors for commentary generation.
    """
    analysis: VisionAnalysisResult
    timestamp: float
    frame_number: int
    image_size: tuple  # (width, height)


class ScreenAnalysisProcessor(FrameProcessor):
    """Processor that analyzes screen captures using vision AI.

    This processor:
    1. Receives UserImageRawFrame from screen capture
    2. Applies rate limiting based on analysis_interval
    3. Sends frames to VisionService for analysis
    4. Emits ScreenAnalysisFrame with results
    5. Passes through all other frames unchanged

    Configuration:
    - analysis_interval: Minimum seconds between analyses
    - cost_limit: Maximum cost per session before stopping
    """

    def __init__(
        self,
        vision_service: VisionService,
        analysis_interval: float = 2.0,
        cost_limit: float = 0.30,
        analysis_prompt: str = None,
        **kwargs
    ):
        """Initialize the screen analysis processor.

        Args:
            vision_service: Configured VisionService for analysis
            analysis_interval: Minimum seconds between analyses
            cost_limit: Maximum cost per session (in USD)
            analysis_prompt: Custom prompt for vision analysis
            **kwargs: Additional FrameProcessor arguments
        """
        super().__init__(**kwargs)

        self.vision_service = vision_service
        self.analysis_interval = analysis_interval
        self.cost_limit = cost_limit
        self.analysis_prompt = analysis_prompt or self._default_prompt()

        # Tracking state
        self._last_analysis_time: float = 0
        self._frame_count: int = 0
        self._total_cost: float = 0.0
        self._analysis_count: int = 0
        self._cost_exceeded: bool = False

        logger.info(
            f"ScreenAnalysisProcessor initialized: "
            f"interval={analysis_interval}s, cost_limit=${cost_limit}"
        )

    def _default_prompt(self) -> str:
        """Return the default analysis prompt for mediation context."""
        return """You are analyzing a screen shared during a co-founder mediation session.

Describe what you see, focusing on content that might be RELEVANT to a business dispute:
- Document type (spreadsheet, contract, email, presentation, code, etc.)
- Key data points, figures, or sections visible
- Any highlighted areas, errors, or points of potential disagreement
- Names, dates, or specific information that could be discussed

Keep your response brief (2-3 sentences). Focus on FACTS visible on screen.
If it's just a desktop or nothing business-relevant, say "Nothing notable".

Remember: This analysis will help an AI mediator reference what's being shown."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames, analyzing screen captures when appropriate.

        Args:
            frame: The frame to process
            direction: Direction of frame flow
        """
        await super().process_frame(frame, direction)

        # Handle screen capture frames
        if isinstance(frame, UserImageRawFrame):
            self._frame_count += 1

            # Check if we should analyze this frame
            if self._should_analyze():
                logger.info(f"📸 Analyzing screen frame #{self._frame_count}")

                try:
                    analysis = await self._analyze_frame(frame)

                    if analysis:
                        # Emit analysis frame
                        analysis_frame = ScreenAnalysisFrame(
                            analysis=analysis,
                            timestamp=time.time(),
                            frame_number=self._frame_count,
                            image_size=self._get_frame_size(frame),
                        )
                        await self.push_frame(analysis_frame, direction)

                        # Update tracking
                        self._last_analysis_time = time.time()
                        self._analysis_count += 1
                        self._total_cost += analysis.cost_usd

                        logger.info(
                            f"✅ Analysis complete: {analysis.description[:80]}... "
                            f"(cost: ${self._total_cost:.4f})"
                        )

                except Exception as e:
                    logger.error(f"Screen analysis failed: {e}", exc_info=True)

        # Always pass through the original frame
        await self.push_frame(frame, direction)

    def _should_analyze(self) -> bool:
        """Determine if current frame should be analyzed.

        Returns:
            True if analysis should proceed, False otherwise
        """
        # Check cost limit
        if self._total_cost >= self.cost_limit:
            if not self._cost_exceeded:
                logger.warning(
                    f"Cost limit exceeded: ${self._total_cost:.4f} >= ${self.cost_limit}. "
                    "Pausing screen analysis."
                )
                self._cost_exceeded = True
            return False

        # Check time interval
        current_time = time.time()
        elapsed = current_time - self._last_analysis_time

        if elapsed < self.analysis_interval:
            return False

        return True

    async def _analyze_frame(self, frame: UserImageRawFrame) -> Optional[VisionAnalysisResult]:
        """Analyze a screen frame using vision service.

        Args:
            frame: UserImageRawFrame with screen capture

        Returns:
            VisionAnalysisResult or None on failure
        """
        try:
            # Extract image data from frame
            image_data = frame.image

            # Get frame metadata if available
            size = self._get_frame_size(frame)
            format_str = getattr(frame, 'format', 'RGB')

            # Perform analysis
            result = await self.vision_service.analyze_image(
                image_data=image_data,
                prompt=self.analysis_prompt,
                format=format_str,
                size=size,
            )

            return result

        except Exception as e:
            logger.error(f"Frame analysis error: {e}", exc_info=True)
            return None

    def _get_frame_size(self, frame: UserImageRawFrame) -> tuple:
        """Get frame dimensions.

        Args:
            frame: UserImageRawFrame

        Returns:
            Tuple of (width, height) or (0, 0) if unavailable
        """
        if hasattr(frame, 'size'):
            return frame.size
        elif hasattr(frame, 'width') and hasattr(frame, 'height'):
            return (frame.width, frame.height)
        return (0, 0)

    def get_stats(self) -> dict:
        """Get processor statistics.

        Returns:
            Dictionary with processing statistics
        """
        return {
            "frames_received": self._frame_count,
            "analyses_performed": self._analysis_count,
            "total_cost_usd": self._total_cost,
            "cost_limit_usd": self.cost_limit,
            "cost_exceeded": self._cost_exceeded,
            "average_interval_actual": (
                (time.time() - self._last_analysis_time) / max(self._analysis_count, 1)
            ) if self._analysis_count > 0 else 0,
        }

    def reset(self):
        """Reset processor state for a new session."""
        logger.info("Resetting ScreenAnalysisProcessor")
        self._last_analysis_time = 0
        self._frame_count = 0
        self._total_cost = 0.0
        self._analysis_count = 0
        self._cost_exceeded = False

    def __repr__(self) -> str:
        return (
            f"ScreenAnalysisProcessor("
            f"interval={self.analysis_interval}s, "
            f"cost_limit=${self.cost_limit})"
        )


def create_screen_analyzer(
    vision_service: VisionService,
    analysis_interval: float = 2.0,
    cost_limit: float = 0.30,
) -> ScreenAnalysisProcessor:
    """Factory function to create a screen analysis processor.

    Args:
        vision_service: Configured VisionService
        analysis_interval: Seconds between analyses
        cost_limit: Maximum session cost in USD

    Returns:
        Configured ScreenAnalysisProcessor
    """
    return ScreenAnalysisProcessor(
        vision_service=vision_service,
        analysis_interval=analysis_interval,
        cost_limit=cost_limit,
    )
