"""
ElevenLabs Text-to-Speech Service for Voice Referee.

Provides ultra-low latency TTS streaming optimized for real-time sports commentary.
Target TTFB: < 300ms using eleven_flash_v2_5 model with WebSocket streaming.
"""

import logging
from typing import Optional

from pipecat.services.elevenlabs import ElevenLabsTTSService

from src.config import TTSConfig

logger = logging.getLogger(__name__)


def create_tts_service(config: TTSConfig) -> ElevenLabsTTSService:
    """
    Create and configure ElevenLabs TTS service for Voice Referee.

    Args:
        config: TTSConfig containing voice settings and API credentials

    Returns:
        Configured ElevenLabsTTSService instance optimized for low latency

    Raises:
        ValueError: If required configuration is missing
        RuntimeError: If service initialization fails

    Configuration:
        - Model: eleven_flash_v2_5 (lowest latency)
        - Voice: Calm, authoritative voice for referee announcements
        - Stability: 0.5 (balanced between consistency and expression)
        - Similarity Boost: 0.75 (high voice similarity)
        - Streaming Latency: Level 4 (maximum optimization)
        - Output Format: PCM 16kHz (Daily.co compatible)

    Performance Targets:
        - Time to First Byte (TTFB): < 300ms
        - Streaming latency: < 500ms end-to-end
        - Audio quality: Clear, natural speech suitable for sports commentary
    """
    try:
        # Validate required configuration
        if not config.api_key:
            raise ValueError("ElevenLabs API key is required")

        if not config.voice_id:
            raise ValueError("Voice ID must be configured")

        logger.info(
            "Initializing ElevenLabs TTS service",
            extra={
                "model": "eleven_flash_v2_5",
                "voice_id": config.voice_id,
                "optimize_streaming_latency": 4,
            }
        )

        # Create service with optimal settings for real-time streaming
        service = ElevenLabsTTSService(
            api_key=config.api_key,
            voice_id=config.voice_id,
            model="eleven_flash_v2_5",  # Fastest model for lowest latency

            # Voice quality settings
            stability=0.5,  # Balanced: not too monotone, not too variable
            similarity_boost=0.75,  # High similarity to selected voice

            # Latency optimization
            optimize_streaming_latency=4,  # Maximum optimization (0-4 scale)

            # Output format compatible with Daily.co
            output_format="pcm_16000",  # 16kHz PCM for Daily.co WebRTC
        )

        logger.info(
            "ElevenLabs TTS service initialized successfully",
            extra={
                "voice_id": config.voice_id,
                "expected_ttfb_ms": "< 300",
                "output_format": "pcm_16000",
            }
        )

        return service

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise

    except Exception as e:
        logger.error(
            "Failed to initialize ElevenLabs TTS service",
            exc_info=True,
            extra={"error": str(e)}
        )
        raise RuntimeError(f"TTS service initialization failed: {e}") from e


class TTSServiceMonitor:
    """
    Monitor TTS service performance and health.

    Tracks metrics for debugging and optimization:
    - Time to first byte (TTFB)
    - Total generation time
    - Audio duration vs processing time ratio
    - Error rates and types
    """

    def __init__(self):
        self.total_generations = 0
        self.total_errors = 0
        self.ttfb_samples = []
        self.generation_times = []

    def log_generation(
        self,
        text: str,
        ttfb_ms: float,
        total_time_ms: float,
        audio_duration_ms: Optional[float] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Log a TTS generation attempt for monitoring.

        Args:
            text: Text that was converted to speech
            ttfb_ms: Time to first byte in milliseconds
            total_time_ms: Total generation time in milliseconds
            audio_duration_ms: Duration of generated audio
            error: Error message if generation failed
        """
        self.total_generations += 1

        if error:
            self.total_errors += 1
            logger.warning(
                "TTS generation failed",
                extra={
                    "text_preview": text[:50],
                    "error": error,
                    "total_errors": self.total_errors,
                }
            )
        else:
            self.ttfb_samples.append(ttfb_ms)
            self.generation_times.append(total_time_ms)

            # Calculate real-time factor if audio duration available
            rtf = None
            if audio_duration_ms and audio_duration_ms > 0:
                rtf = total_time_ms / audio_duration_ms

            logger.debug(
                "TTS generation completed",
                extra={
                    "text_preview": text[:50],
                    "ttfb_ms": round(ttfb_ms, 2),
                    "total_time_ms": round(total_time_ms, 2),
                    "audio_duration_ms": round(audio_duration_ms, 2) if audio_duration_ms else None,
                    "real_time_factor": round(rtf, 3) if rtf else None,
                    "target_met": ttfb_ms < 300,
                }
            )

    def get_stats(self) -> dict:
        """
        Get performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        if not self.ttfb_samples:
            return {
                "total_generations": self.total_generations,
                "total_errors": self.total_errors,
                "error_rate": self.total_errors / max(self.total_generations, 1),
            }

        return {
            "total_generations": self.total_generations,
            "total_errors": self.total_errors,
            "error_rate": self.total_errors / self.total_generations,
            "avg_ttfb_ms": sum(self.ttfb_samples) / len(self.ttfb_samples),
            "max_ttfb_ms": max(self.ttfb_samples),
            "min_ttfb_ms": min(self.ttfb_samples),
            "avg_generation_ms": sum(self.generation_times) / len(self.generation_times),
            "ttfb_target_met_pct": sum(1 for t in self.ttfb_samples if t < 300) / len(self.ttfb_samples) * 100,
        }


# Global monitor instance for service-wide metrics
tts_monitor = TTSServiceMonitor()


def get_tts_stats() -> dict:
    """
    Get current TTS service statistics.

    Returns:
        Performance metrics dictionary
    """
    return tts_monitor.get_stats()
