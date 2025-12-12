"""
Referee Monitor Processor - Main orchestration processor

Receives transcription frames from Deepgram, analyzes conversation state,
and triggers AI referee interventions when needed.
"""

import logging
import time
from typing import Optional

from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    LLMMessagesFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    TextFrame
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from .speaker_mapper import SpeakerMapper
from .conversation_state import ConversationState
from ..analysis.conversation_analyzer import ConversationAnalyzer
from ..decision.intervention_decider import InterventionDecider
from ..config.settings import ProcessorConfig

logger = logging.getLogger(__name__)


class RefereeMonitorProcessor(FrameProcessor):
    """
    Main processor that orchestrates the Voice Referee system.

    Flow:
    1. Receives TranscriptionFrame from Deepgram (with speaker diarization)
    2. Maps speaker ID to founder name via SpeakerMapper
    3. Updates ConversationState with utterance
    4. Runs ConversationAnalyzer to analyze conversation health
    5. Checks InterventionDecider for intervention recommendation
    6. If needed: creates LLMMessagesFrame with intervention context
    7. Passes frames downstream for LLM processing and TTS
    """

    def __init__(
        self,
        config: Optional[ProcessorConfig] = None,
        founder_names: Optional[list[str]] = None,
        **kwargs
    ):
        """
        Initialize the referee monitor processor.

        Args:
            config: Processor configuration (uses defaults if None)
            founder_names: Optional list of founder names for speaker mapping
            **kwargs: Additional arguments for FrameProcessor
        """
        super().__init__(**kwargs)

        # Use provided config or create default
        self._config = config or ProcessorConfig()

        # Initialize components
        self._speaker_mapper = SpeakerMapper(founder_names=founder_names)
        self._conversation_state = ConversationState(
            max_buffer_size=self._config.buffer_size
        )
        self._analyzer = ConversationAnalyzer(
            tension_threshold=self._config.tension_threshold
        )
        self._decider = InterventionDecider(
            cooldown_seconds=self._config.cooldown_seconds
        )

        # Track speaking state for duration calculation
        self._current_speaker: Optional[str] = None
        self._speaking_start_time: Optional[float] = None

        logger.info(
            f"RefereeMonitorProcessor initialized with config: "
            f"tension_threshold={self._config.tension_threshold}, "
            f"cooldown={self._config.cooldown_seconds}s, "
            f"buffer_size={self._config.buffer_size}"
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """
        Process incoming frames.

        Handles:
        - TranscriptionFrame: Extract speaker, update state, check for intervention
        - UserStartedSpeakingFrame: Track speaking start time
        - UserStoppedSpeakingFrame: Calculate speaking duration
        - Other frames: Pass through unchanged

        Args:
            frame: The frame to process
            direction: Direction of frame flow
        """
        await super().process_frame(frame, direction)

        # Handle transcription frames with speaker diarization
        if isinstance(frame, TranscriptionFrame):
            await self._handle_transcription(frame)

        # Handle user speaking state changes
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._handle_speaking_started()

        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._handle_speaking_stopped()

        # Pass frame downstream
        await self.push_frame(frame, direction)

    async def _handle_transcription(self, frame: TranscriptionFrame):
        """
        Handle transcription frame from Deepgram.

        Args:
            frame: TranscriptionFrame with text and optional speaker attribute
        """
        try:
            # Extract transcription text
            text = frame.text.strip()
            if not text:
                logger.debug("Empty transcription, skipping")
                return

            # Get speaker ID from frame (Deepgram provides this)
            # The speaker attribute should be set by Deepgram STT service
            speaker_id = getattr(frame, 'speaker', None)

            if speaker_id is None:
                logger.warning("TranscriptionFrame missing 'speaker' attribute, skipping")
                return

            # Map speaker ID to founder name
            founder_name = self._speaker_mapper.assign_identity(speaker_id)

            # Calculate duration (use tracked time or estimate from text)
            duration = self._calculate_duration(text)

            # Add utterance to conversation state
            timestamp = time.time()
            self._conversation_state.add_utterance(
                text=text,
                speaker=founder_name,
                timestamp=timestamp,
                duration=duration
            )

            logger.info(f"Utterance recorded: {founder_name}: {text[:50]}...")

            # Analyze conversation
            logger.info("Running conversation analysis...")
            analysis = self._analyzer.analyze(self._conversation_state)
            logger.info(f"Analysis complete: tension={analysis.tension_score:.2f}, requires_intervention={analysis.requires_intervention}")

            # Check if intervention is needed
            decision = self._decider.decide(analysis, self._conversation_state)
            logger.info(f"Decision: should_intervene={decision.should_intervene}, reason={decision.reason}")

            if decision.should_intervene:
                await self._trigger_intervention(decision, analysis)

        except Exception as e:
            logger.error(f"Error handling transcription: {e}", exc_info=True)

    async def _trigger_intervention(self, decision, analysis):
        """
        Trigger an AI referee intervention.

        Args:
            decision: InterventionDecision with intervention details
            analysis: AnalysisResult with conversation analysis
        """
        logger.info(
            f"🚨 INTERVENTION TRIGGERED - {decision.reason} "
            f"(confidence={decision.confidence:.2f})"
        )

        # Build system message for Claude
        system_prompt = """You are an AI referee moderating a conversation between startup founders.
Your role is to:
- Ensure balanced participation
- De-escalate tension
- Keep the conversation productive
- Be brief, diplomatic, and constructive

When you intervene, keep your message concise (2-3 sentences) and supportive."""

        # Create LLM messages frame with intervention context
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": decision.suggested_prompt
            }
        ]

        # Push LLMMessagesFrame to trigger LLM response
        llm_frame = LLMMessagesFrame(messages=messages)
        await self.push_frame(llm_frame)

        # Record intervention in both decider and state
        self._decider.record_intervention()
        self._conversation_state.record_intervention()

        # Log intervention details
        logger.info(f"Intervention context: {decision.suggested_prompt[:200]}...")
        logger.info(f"Analysis: tension={analysis.tension_score:.2f}, patterns={analysis.detected_patterns}")

    def _handle_speaking_started(self):
        """Handle when a user starts speaking."""
        self._speaking_start_time = time.time()
        logger.debug("User started speaking")

    def _handle_speaking_stopped(self):
        """Handle when a user stops speaking."""
        if self._speaking_start_time:
            duration = time.time() - self._speaking_start_time
            logger.debug(f"User stopped speaking (duration: {duration:.2f}s)")
            self._speaking_start_time = None

    def _calculate_duration(self, text: str) -> float:
        """
        Calculate/estimate utterance duration.

        Args:
            text: Transcribed text

        Returns:
            Duration in seconds
        """
        # If we tracked speaking time, use it
        if self._speaking_start_time:
            duration = time.time() - self._speaking_start_time
            self._speaking_start_time = None
            return duration

        # Otherwise estimate based on word count (average 150 words/minute)
        word_count = len(text.split())
        estimated_duration = (word_count / 150.0) * 60.0
        return max(estimated_duration, 0.5)  # Minimum 0.5 seconds

    def get_state(self) -> ConversationState:
        """Get current conversation state."""
        return self._conversation_state

    def get_stats(self) -> dict:
        """Get comprehensive conversation statistics."""
        return self._conversation_state.get_stats()

    def reset(self):
        """Reset all components for a new session."""
        logger.info("Resetting referee monitor for new session")
        self._speaker_mapper.reset()
        self._conversation_state.reset()
        self._decider.reset_cooldown()
        self._current_speaker = None
        self._speaking_start_time = None

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RefereeMonitorProcessor("
            f"tension_threshold={self._config.tension_threshold}, "
            f"cooldown={self._config.cooldown_seconds}s)"
        )
