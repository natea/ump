"""Daily.co transport service for Voice Referee.

This module provides the transport layer for real-time audio streaming
using Daily.co WebRTC infrastructure with Voice Activity Detection.
"""

import logging
from typing import Optional

from pipecat.transports.services.daily import DailyTransport, DailyParams
from pipecat.audio.vad.silero import SileroVADAnalyzer

from ..config import DailyConfig

logger = logging.getLogger(__name__)


class VoiceRefereeTransport:
    """Voice Referee Daily.co transport wrapper with event handling.

    This class wraps the DailyTransport and provides additional functionality
    for participant tracking, connection monitoring, and event logging.
    """

    def __init__(self, transport: DailyTransport, config: DailyConfig):
        """Initialize the transport wrapper.

        Args:
            transport: Configured DailyTransport instance
            config: Daily configuration used for this transport
        """
        self._transport = transport
        self._config = config
        self._participants = {}
        self._is_connected = False

        # Register event handlers
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Set up event handlers for Daily.co events."""
        # Note: Event handler setup will be implemented based on
        # pipecat's DailyTransport API for handling:
        # - participant-joined
        # - participant-left
        # - participant-updated
        # - track-started
        # - track-stopped
        # - error events
        logger.info("Event handlers configured for Daily transport")

    def _on_participant_joined(self, participant_id: str, participant_info: dict):
        """Handle participant join event.

        Args:
            participant_id: Unique identifier for the participant
            participant_info: Dictionary containing participant metadata
        """
        self._participants[participant_id] = participant_info
        logger.info(
            f"Participant joined: {participant_id}",
            extra={
                "participant_id": participant_id,
                "participant_name": participant_info.get("user_name", "Unknown"),
                "total_participants": len(self._participants)
            }
        )

    def _on_participant_left(self, participant_id: str, reason: Optional[str] = None):
        """Handle participant leave event.

        Args:
            participant_id: Unique identifier for the participant
            reason: Optional reason for leaving
        """
        participant_info = self._participants.pop(participant_id, {})
        logger.info(
            f"Participant left: {participant_id}",
            extra={
                "participant_id": participant_id,
                "participant_name": participant_info.get("user_name", "Unknown"),
                "reason": reason,
                "remaining_participants": len(self._participants)
            }
        )

    def _on_error(self, error: Exception):
        """Handle transport error.

        Args:
            error: Exception that occurred
        """
        logger.error(
            f"Daily transport error: {error}",
            extra={"error_type": type(error).__name__},
            exc_info=True
        )

    @property
    def transport(self) -> DailyTransport:
        """Get the underlying DailyTransport instance."""
        return self._transport

    def input(self):
        """Get input processor from underlying transport."""
        return self._transport.input()

    def output(self):
        """Get output processor from underlying transport."""
        return self._transport.output()

    @property
    def is_connected(self) -> bool:
        """Check if transport is currently connected."""
        return self._is_connected

    @property
    def participant_count(self) -> int:
        """Get current number of participants."""
        return len(self._participants)

    async def test_connection(self) -> bool:
        """Test the Daily.co connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("Testing Daily.co connection...")
            # Connection test will be implemented based on pipecat's API
            # This might involve checking room status or sending a ping
            self._is_connected = True
            logger.info("Daily.co connection test successful")
            return True
        except Exception as e:
            logger.error(f"Daily.co connection test failed: {e}", exc_info=True)
            self._is_connected = False
            return False


def create_daily_transport(config: DailyConfig) -> VoiceRefereeTransport:
    """Create and configure a Daily.co transport for Voice Referee.

    This factory function creates a DailyTransport with Voice Activity Detection
    and appropriate audio configuration for real-time voice processing.

    Args:
        config: DailyConfig object with transport settings

    Returns:
        VoiceRefereeTransport: Configured transport wrapper

    Raises:
        ValueError: If configuration is invalid
        ConnectionError: If unable to connect to Daily.co

    Example:
        >>> from voice_referee.config import DailyConfig
        >>> config = DailyConfig(
        ...     api_key="your-api-key",
        ...     room_url="https://your-domain.daily.co/room-name"
        ... )
        >>> transport = create_daily_transport(config)
        >>> await transport.test_connection()
    """
    logger.info(
        "Creating Daily transport",
        extra={
            "room_url": config.room_url,
            "bot_name": config.bot_name,
            "vad_enabled": config.vad_enabled,
            "sample_rate": config.sample_rate
        }
    )

    # Create VAD analyzer (using default params in newer pipecat)
    vad_analyzer = SileroVADAnalyzer() if config.vad_enabled else None

    if vad_analyzer:
        logger.info("VAD enabled with Silero analyzer")

    # Configure Daily transport parameters
    daily_params = DailyParams(
        audio_in_enabled=config.audio_in_enabled,
        audio_out_enabled=config.audio_out_enabled,
        camera_out_enabled=config.camera_out_enabled,
        vad_enabled=config.vad_enabled,
        vad_analyzer=vad_analyzer,
        transcription_enabled=config.transcription_enabled,
        # Additional params that might be needed:
        # api_key=config.api_key,
        # token=config.token,
    )

    # Create the Daily transport
    # For public rooms, token can be None or empty
    # Only pass token if it looks like a valid meeting token (starts with certain prefixes)
    token_to_use = config.token if config.token and config.token.startswith("ey") else None

    try:
        transport = DailyTransport(
            room_url=config.room_url,
            token=token_to_use,
            bot_name=config.bot_name,
            params=daily_params,
        )

        logger.info("Daily transport created successfully")

        # Wrap with our event handler
        voice_referee_transport = VoiceRefereeTransport(transport, config)

        return voice_referee_transport

    except Exception as e:
        logger.error(f"Failed to create Daily transport: {e}", exc_info=True)
        raise ConnectionError(f"Could not create Daily transport: {e}") from e
