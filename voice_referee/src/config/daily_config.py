"""Daily.co configuration for Voice Referee."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DailyConfig:
    """Configuration for Daily.co transport.

    Attributes:
        api_key: Daily.co API key for authentication
        room_url: Daily.co room URL to join
        token: Optional Daily.co meeting token for authentication
        bot_name: Name identifier for the bot participant
        sample_rate: Audio sample rate in Hz (default: 16000)
        audio_in_enabled: Enable audio input from participants
        audio_out_enabled: Enable audio output to participants
        camera_out_enabled: Enable camera output (disabled for voice-only)
        vad_enabled: Enable Voice Activity Detection
        vad_min_volume: Minimum volume threshold for VAD (0.0-1.0)
        transcription_enabled: Enable Daily's built-in transcription (we use Deepgram)
    """

    api_key: str
    room_url: str
    token: Optional[str] = None
    bot_name: str = "Voice Referee"
    sample_rate: int = 16000
    audio_in_enabled: bool = True
    audio_out_enabled: bool = True
    camera_out_enabled: bool = False
    vad_enabled: bool = True
    vad_min_volume: float = 0.6
    transcription_enabled: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.api_key:
            raise ValueError("Daily API key is required")
        if not self.room_url:
            raise ValueError("Daily room URL is required")
        if not 0.0 <= self.vad_min_volume <= 1.0:
            raise ValueError("VAD min volume must be between 0.0 and 1.0")
        if self.sample_rate not in [8000, 16000, 24000, 48000]:
            raise ValueError("Sample rate must be 8000, 16000, 24000, or 48000 Hz")
