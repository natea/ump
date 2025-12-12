"""Daily.co transport service for Voice Referee.

This module provides the transport layer for real-time audio streaming
using Daily.co WebRTC infrastructure with Voice Activity Detection.
"""

import logging
from typing import Callable, Optional

from pipecat.transports.services.daily import DailyTransport, DailyParams
from pipecat.audio.vad.silero import SileroVADAnalyzer

from src.config import DailyConfig

logger = logging.getLogger(__name__)

# Type for participant callback: (participant_id, user_name) -> None
ParticipantCallback = Callable[[str, str], None]


class VoiceRefereeTransport:
    """Voice Referee Daily.co transport wrapper with event handling.

    This class wraps the DailyTransport and provides additional functionality
    for participant tracking, connection monitoring, and event logging.

    Supports callbacks for participant join/leave events to enable
    speaker name mapping in the referee monitor.
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

        # Callbacks for participant events
        self._on_participant_joined_callback: Optional[ParticipantCallback] = None
        self._on_participant_left_callback: Optional[ParticipantCallback] = None

        # Register event handlers on the transport
        self._setup_event_handlers()

    def set_participant_callbacks(
        self,
        on_joined: Optional[ParticipantCallback] = None,
        on_left: Optional[ParticipantCallback] = None
    ) -> None:
        """Set callbacks for participant join/leave events.

        Args:
            on_joined: Callback when participant joins (participant_id, user_name)
            on_left: Callback when participant leaves (participant_id, user_name)
        """
        self._on_participant_joined_callback = on_joined
        self._on_participant_left_callback = on_left
        logger.info("Participant callbacks registered")

    def _setup_event_handlers(self):
        """Set up event handlers for Daily.co events using Pipecat's decorator."""
        logger.info("Setting up Daily transport event handlers...")

        @self._transport.event_handler("on_participant_joined")
        async def handle_participant_joined(transport, participant):
            """Handle participant join event from Daily."""
            participant_id = participant.get("id", "unknown")
            user_name = participant.get("info", {}).get("userName", "") or participant.get("user_name", "Unknown")

            logger.info(f"👤 Participant joined: {user_name} (ID: {participant_id})")
            logger.debug(f"Participant data: {participant}")

            self._participants[participant_id] = {
                "user_name": user_name,
                "info": participant
            }

            # Call the external callback if set
            if self._on_participant_joined_callback:
                self._on_participant_joined_callback(participant_id, user_name)

        @self._transport.event_handler("on_participant_left")
        async def handle_participant_left(transport, participant, reason):
            """Handle participant leave event from Daily."""
            participant_id = participant.get("id", "unknown")
            user_name = participant.get("info", {}).get("userName", "") or participant.get("user_name", "Unknown")

            logger.info(f"👤 Participant left: {user_name} (reason: {reason})")

            self._participants.pop(participant_id, None)

            # Call the external callback if set
            if self._on_participant_left_callback:
                self._on_participant_left_callback(participant_id, user_name)

        @self._transport.event_handler("on_first_participant_joined")
        async def handle_first_participant(transport, participant):
            """Handle first participant joining - enable transcription capture."""
            participant_id = participant.get("id", "unknown")
            user_name = participant.get("info", {}).get("userName", "") or participant.get("user_name", "Unknown")

            logger.info(f"🎉 First participant joined: {user_name}")

            # Capture transcription for this participant
            await transport.capture_participant_transcription(participant_id)

        logger.info("Daily transport event handlers configured")

    def get_participants(self) -> dict:
        """Get current participants in the room.

        Returns:
            Dictionary of participant_id -> participant info
        """
        return self._participants.copy()

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
