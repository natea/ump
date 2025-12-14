"""
Commentary Processor - Generates spoken commentary from screen analysis.

This processor receives ScreenAnalysisFrame from the screen analyzer
and generates natural language commentary to be spoken via TTS.
"""

import logging
from typing import Optional

from pipecat.frames.frames import Frame, LLMMessagesFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

# Import will work once screen_analyzer.py is created
# from src.processors.screen_analyzer import ScreenAnalysisFrame

logger = logging.getLogger(__name__)


# System prompts for different commentary styles
# These are used by the AI mediator when commenting on shared screen content
COMMENTARY_PROMPTS = {
    "concise": """You are an AI mediator who can see the participants' shared screen.

CONTEXT: Two co-founders are in a mediation session. One of them is sharing their screen.

Guidelines:
- Keep responses SHORT (1-2 sentences max)
- ONLY comment if the screen content is RELEVANT to the discussion
- Reference what you see naturally: "I see the spreadsheet you mentioned..."
- Ask clarifying questions: "Is this the revenue data you were discussing?"
- Help connect visual content to the conversation
- If nothing relevant, return [SKIP] - don't comment on every screen change

Good examples:
- "I can see the document you're referring to. Which section concerns you most?"
- "This spreadsheet shows the figures you mentioned—the Q3 column looks significant."
- "I notice there's an error highlighted here. Is this what you wanted to discuss?"

BAD examples (too generic):
- "I see a spreadsheet on the screen." (So what? How does it help?)
- "You're showing a document." (Obvious and unhelpful)

Remember: You're a MEDIATOR, not a screen narrator. Only speak if it helps the discussion.""",

    "detailed": """You are an AI mediator who can see the participants' shared screen.

CONTEXT: Two co-founders are in a mediation session discussing a dispute. One is sharing their screen.

Guidelines:
- Provide 2-3 sentences connecting what you SEE to what they're DISCUSSING
- Describe key elements that are relevant to the conflict
- Note specific data points, sections, or issues visible
- Help both parties focus on the same information
- Ask questions that help clarify disagreements
- If nothing relevant to the discussion, return [SKIP]

Your role: Use the screen to help facilitate understanding between both parties.

Remember: This is VOICE output. Be clear and helpful, not just descriptive.""",

    "technical": """You are an AI mediator with technical expertise who can see shared screen content.

CONTEXT: Two co-founders in a mediation session are discussing technical/business matters.

Guidelines:
- Focus on technical details RELEVANT to their dispute
- Identify specific data, code, or documents that relate to their discussion
- Note discrepancies or issues that might be causing conflict
- Provide observations that help clarify technical disagreements
- 2-3 sentences maximum
- If nothing relevant, return [SKIP]

Help them get on the same page by pointing out what the data/documents actually show.

Remember: Technical accuracy matters, but your job is mediation, not just analysis.""",
}


class CommentaryProcessor(FrameProcessor):
    """Processor that generates spoken commentary from screen analysis.

    This processor:
    1. Receives ScreenAnalysisFrame with vision analysis results
    2. Filters based on commentary requirements
    3. Generates natural language commentary via LLM
    4. Emits LLMMessagesFrame for TTS processing
    """

    def __init__(
        self,
        commentary_style: str = "concise",
        trigger_threshold: float = 0.6,
        cooldown_seconds: float = 5.0,
        **kwargs
    ):
        """Initialize the commentary processor.

        Args:
            commentary_style: Style of commentary (concise, detailed, technical)
            trigger_threshold: Confidence threshold for commentary
            cooldown_seconds: Minimum seconds between commentaries
            **kwargs: Additional FrameProcessor arguments
        """
        super().__init__(**kwargs)

        self.commentary_style = commentary_style
        self.trigger_threshold = trigger_threshold
        self.cooldown_seconds = cooldown_seconds

        # Get system prompt for style
        self.system_prompt = COMMENTARY_PROMPTS.get(
            commentary_style,
            COMMENTARY_PROMPTS["concise"]
        )

        # Tracking
        self._last_commentary_time: float = 0
        self._commentary_count: int = 0
        self._skipped_count: int = 0

        logger.info(
            f"CommentaryProcessor initialized: "
            f"style={commentary_style}, threshold={trigger_threshold}"
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, generating commentary for screen analysis.

        Args:
            frame: The frame to process
            direction: Direction of frame flow
        """
        await super().process_frame(frame, direction)

        # Check if this is a screen analysis frame
        # Using duck typing to avoid circular imports
        if hasattr(frame, 'analysis') and hasattr(frame, 'frame_number'):
            await self._handle_analysis_frame(frame, direction)
        else:
            # Pass through all other frames
            await self.push_frame(frame, direction)

    async def _handle_analysis_frame(self, frame, direction: FrameDirection):
        """Handle a screen analysis frame.

        Args:
            frame: ScreenAnalysisFrame with analysis results
            direction: Frame direction
        """
        import time

        analysis = frame.analysis

        # Check if commentary is needed
        if not self._should_generate_commentary(analysis):
            self._skipped_count += 1
            logger.debug(f"Skipping commentary (reason: filtering)")
            await self.push_frame(frame, direction)
            return

        # Check cooldown
        current_time = time.time()
        elapsed = current_time - self._last_commentary_time
        if elapsed < self.cooldown_seconds:
            logger.debug(f"Skipping commentary (cooldown: {self.cooldown_seconds - elapsed:.1f}s remaining)")
            await self.push_frame(frame, direction)
            return

        logger.info(f"💬 Generating commentary for frame #{frame.frame_number}")

        # Build user prompt from analysis
        user_prompt = self._build_user_prompt(analysis)

        # Create LLM messages frame
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        llm_frame = LLMMessagesFrame(messages=messages)
        await self.push_frame(llm_frame, direction)

        # Update tracking
        self._last_commentary_time = current_time
        self._commentary_count += 1

        logger.info(f"✅ Commentary triggered (total: {self._commentary_count})")

        # Pass through the original frame too
        await self.push_frame(frame, direction)

    def _should_generate_commentary(self, analysis) -> bool:
        """Determine if analysis warrants commentary.

        Args:
            analysis: VisionAnalysisResult

        Returns:
            True if commentary should be generated
        """
        # Check if analysis explicitly requires commentary
        if hasattr(analysis, 'requires_commentary') and analysis.requires_commentary:
            return True

        # Check confidence threshold
        if hasattr(analysis, 'confidence'):
            if analysis.confidence < self.trigger_threshold:
                return False

        # Check if description indicates nothing notable
        description = getattr(analysis, 'description', '')
        nothing_phrases = ['nothing notable', 'no changes', 'same as before', 'analysis failed']
        for phrase in nothing_phrases:
            if phrase.lower() in description.lower():
                return False

        return True

    def _build_user_prompt(self, analysis) -> str:
        """Build user prompt from analysis results.

        Args:
            analysis: VisionAnalysisResult

        Returns:
            User prompt string
        """
        description = getattr(analysis, 'description', 'Unknown content')
        confidence = getattr(analysis, 'confidence', 0)
        detected_elements = getattr(analysis, 'detected_elements', [])

        prompt = f"Screen content analysis:\n{description}\n\n"

        if detected_elements:
            prompt += f"Detected elements: {', '.join(detected_elements)}\n\n"

        prompt += "Provide brief spoken commentary about what you observe. "
        prompt += "If nothing is particularly notable, respond with just: [SKIP]"

        return prompt

    def get_stats(self) -> dict:
        """Get processor statistics.

        Returns:
            Dictionary with commentary statistics
        """
        return {
            "commentaries_generated": self._commentary_count,
            "frames_skipped": self._skipped_count,
            "total_frames_processed": self._commentary_count + self._skipped_count,
            "commentary_style": self.commentary_style,
            "trigger_threshold": self.trigger_threshold,
        }

    def reset(self):
        """Reset processor state for a new session."""
        logger.info("Resetting CommentaryProcessor")
        self._last_commentary_time = 0
        self._commentary_count = 0
        self._skipped_count = 0

    def __repr__(self) -> str:
        return (
            f"CommentaryProcessor("
            f"style={self.commentary_style}, "
            f"threshold={self.trigger_threshold})"
        )


def create_commentary_processor(
    commentary_style: str = "concise",
    trigger_threshold: float = 0.6,
    cooldown_seconds: float = 5.0,
) -> CommentaryProcessor:
    """Factory function to create a commentary processor.

    Args:
        commentary_style: Style (concise, detailed, technical)
        trigger_threshold: Confidence threshold for commentary
        cooldown_seconds: Minimum seconds between commentaries

    Returns:
        Configured CommentaryProcessor
    """
    return CommentaryProcessor(
        commentary_style=commentary_style,
        trigger_threshold=trigger_threshold,
        cooldown_seconds=cooldown_seconds,
    )
