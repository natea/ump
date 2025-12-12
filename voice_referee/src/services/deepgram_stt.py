"""Deepgram Speech-to-Text service with diarization support.

This module provides a factory function to create a Deepgram STT service
configured for mediation sessions with speaker diarization.

HIGH RISK: Diarization accuracy is critical for distinguishing between
the two founders. Comprehensive logging is included for debugging.
"""

import logging
from typing import Optional

from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.frames.frames import TranscriptionFrame, InterimTranscriptionFrame
from pipecat.utils.time import time_now_iso8601
from pipecat.transcriptions.language import Language

from deepgram import LiveOptions

# Configure logging
logger = logging.getLogger(__name__)


class DeepgramConfig:
    """Configuration for Deepgram STT service.

    Attributes:
        api_key: Deepgram API key for authentication
        model: Deepgram model to use (default: nova-2)
        language: Language code (default: en)
        diarize: Enable speaker diarization (default: True)
        punctuate: Enable automatic punctuation (default: True)
        interim_results: Enable interim transcription results (default: True)
        smart_format: Enable smart formatting (default: True)
        utterance_end_ms: Milliseconds of silence to mark utterance end (default: 1000)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "nova-2",
        language: str = "en",
        diarize: bool = True,
        punctuate: bool = True,
        interim_results: bool = True,
        smart_format: bool = True,
        utterance_end_ms: int = 1000,
    ):
        self.api_key = api_key
        self.model = model
        self.language = language
        self.diarize = diarize
        self.punctuate = punctuate
        self.interim_results = interim_results
        self.smart_format = smart_format
        self.utterance_end_ms = utterance_end_ms

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid
        """
        if not self.api_key:
            raise ValueError("Deepgram API key is required")

        if self.utterance_end_ms < 100:
            raise ValueError("utterance_end_ms must be at least 100ms")

        if self.utterance_end_ms > 10000:
            logger.warning(
                "utterance_end_ms is very high (%dms). This may cause delays.",
                self.utterance_end_ms
            )


class DiarizedTranscriptionFrame(TranscriptionFrame):
    """Extended TranscriptionFrame that includes speaker diarization info."""

    def __init__(
        self,
        text: str,
        user_id: str,
        timestamp: str,
        language: Optional[Language] = None,
        result=None,
        speaker: Optional[int] = None,
    ):
        super().__init__(text, user_id, timestamp, language, result=result)
        self.speaker = speaker


class DiarizedDeepgramSTTService(DeepgramSTTService):
    """Extended Deepgram STT service with speaker diarization support.

    This class extends the base DeepgramSTTService to extract speaker
    information from Deepgram's diarization results and attach it to
    TranscriptionFrames.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the diarized STT service."""
        super().__init__(*args, **kwargs)
        self._transcription_count = 0
        self._speaker_stats = {}
        logger.info("DiarizedDeepgramSTTService initialized with diarization enabled")

    async def _on_message(self, *args, **kwargs):
        """Handle incoming Deepgram transcription messages with speaker extraction.

        Overrides the parent method to extract speaker ID from diarization
        results and attach it to the TranscriptionFrame.
        """
        result = kwargs.get("result")
        if not result or len(result.channel.alternatives) == 0:
            return

        is_final = result.is_final
        transcript = result.channel.alternatives[0].transcript

        # Extract language if available
        language = None
        if result.channel.alternatives[0].languages:
            language = result.channel.alternatives[0].languages[0]
            language = Language(language)

        # Extract speaker from diarization results
        speaker = None
        words = getattr(result.channel.alternatives[0], 'words', [])
        if words:
            # Get the most common speaker in this segment
            speaker_counts = {}
            for word in words:
                word_speaker = getattr(word, 'speaker', None)
                if word_speaker is not None:
                    speaker_counts[word_speaker] = speaker_counts.get(word_speaker, 0) + 1

            if speaker_counts:
                # Use the speaker who spoke the most words in this segment
                speaker = max(speaker_counts, key=speaker_counts.get)
                logger.debug(f"Extracted speaker {speaker} from diarization (counts: {speaker_counts})")

        if len(transcript) > 0:
            await self.stop_ttfb_metrics()

            self._transcription_count += 1

            if is_final:
                # Update speaker statistics
                if speaker is not None:
                    self._speaker_stats[speaker] = self._speaker_stats.get(speaker, 0) + 1

                # Create frame with speaker attribute
                frame = DiarizedTranscriptionFrame(
                    transcript,
                    self._user_id,
                    time_now_iso8601(),
                    language,
                    result=result,
                    speaker=speaker,
                )
                await self.push_frame(frame)

                # Log transcription with speaker info
                logger.info(
                    "Transcription #%d [Speaker %s] FINAL: %s",
                    self._transcription_count,
                    speaker if speaker is not None else "UNKNOWN",
                    transcript[:100] + "..." if len(transcript) > 100 else transcript
                )

                if speaker is None:
                    logger.warning(
                        "Missing speaker ID in final transcription. "
                        "Diarization may not be working correctly."
                    )

                await self._handle_transcription(transcript, is_final, language)
                await self.stop_processing_metrics()
            else:
                # For interim transcriptions - still add speaker if available
                frame = InterimTranscriptionFrame(
                    transcript,
                    self._user_id,
                    time_now_iso8601(),
                    language,
                    result=result,
                )
                # Add speaker attribute to interim frame too
                frame.speaker = speaker
                await self.push_frame(frame)

                logger.debug(
                    "Transcription #%d [Speaker %s] interim: %s",
                    self._transcription_count,
                    speaker if speaker is not None else "?",
                    transcript[:50] + "..." if len(transcript) > 50 else transcript
                )

    def get_statistics(self) -> dict:
        """Get service statistics for monitoring.

        Returns:
            Dictionary containing transcription and speaker statistics
        """
        return {
            'total_transcriptions': self._transcription_count,
            'speaker_counts': dict(self._speaker_stats),
            'unique_speakers': len(self._speaker_stats)
        }

    def log_statistics(self) -> None:
        """Log current service statistics."""
        stats = self.get_statistics()
        logger.info(
            "Deepgram STT Statistics - Total: %d, Speakers: %s, Unique: %d",
            stats['total_transcriptions'],
            stats['speaker_counts'],
            stats['unique_speakers']
        )


def create_deepgram_stt(config: DeepgramConfig) -> DiarizedDeepgramSTTService:
    """Create a Deepgram STT service configured for mediation transcription.

    This factory function creates a Deepgram STT service with optimal settings
    for real-time mediation session voice input with speaker diarization to
    distinguish between two founders.

    CRITICAL: Diarization must be enabled to identify which founder is speaking.
    The service will output DiarizedTranscriptionFrame objects with a 'speaker'
    attribute (typically 0 or 1) to identify the speaker.

    Args:
        config: DeepgramConfig instance with API key and optional overrides

    Returns:
        Configured DiarizedDeepgramSTTService instance

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate configuration
    config.validate()

    logger.info("Creating Deepgram STT service with configuration:")
    logger.info("  Model: %s", config.model)
    logger.info("  Language: %s", config.language)
    logger.info("  Diarization: %s (CRITICAL)", config.diarize)
    logger.info("  Punctuation: %s", config.punctuate)
    logger.info("  Interim results: %s", config.interim_results)
    logger.info("  Smart format: %s", config.smart_format)
    logger.info("  Utterance end: %dms", config.utterance_end_ms)

    # Verify diarization is enabled
    if not config.diarize:
        logger.error("CRITICAL: Diarization is disabled! Speaker identification will fail.")
        raise ValueError(
            "Diarization must be enabled for Voice Referee system. "
            "Set diarize=True in DeepgramConfig."
        )

    # Create LiveOptions for detailed configuration
    live_options = LiveOptions(
        model=config.model,
        language=config.language,
        diarize=config.diarize,
        punctuate=config.punctuate,
        interim_results=config.interim_results,
        smart_format=config.smart_format,
        utterance_end_ms=config.utterance_end_ms,
    )

    # Create service with configuration
    try:
        service = DiarizedDeepgramSTTService(
            api_key=config.api_key,
            live_options=live_options,
        )

        logger.info("Deepgram STT service created successfully")
        logger.warning(
            "HIGH RISK: Monitor diarization accuracy. "
            "Speaker misidentification will cause incorrect attribution."
        )

        return service

    except Exception as e:
        logger.error("Failed to create Deepgram STT service: %s", str(e))
        raise


def create_default_stt(api_key: str) -> DiarizedDeepgramSTTService:
    """Create a Deepgram STT service with default configuration.

    This is a convenience function that uses the recommended default
    settings for mediation transcription.

    Args:
        api_key: Deepgram API key

    Returns:
        Configured DiarizedDeepgramSTTService instance
    """
    config = DeepgramConfig(api_key=api_key)
    return create_deepgram_stt(config)


# Export public API
__all__ = [
    'DeepgramConfig',
    'DiarizedDeepgramSTTService',
    'DiarizedTranscriptionFrame',
    'create_deepgram_stt',
    'create_default_stt',
]
