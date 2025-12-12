"""
Shared fixtures for integration tests.

Provides mocks for external services:
- Daily.co transport
- Deepgram STT
- Anthropic Claude LLM
- ElevenLabs TTS
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, MagicMock, patch
from typing import List, Dict, Any
import time

from pipecat.frames.frames import (
    TranscriptionFrame,
    TextFrame,
    LLMMessagesFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame
)


# =====================================================================
# Mock Deepgram STT Service
# =====================================================================

@pytest.fixture
def mock_deepgram_stt():
    """
    Mock Deepgram STT service that simulates speaker diarization.

    Returns a mock service with configurable transcription responses.
    """
    mock_service = AsyncMock()
    mock_service.process_frame = AsyncMock()
    mock_service.get_statistics = Mock(return_value={
        'total_transcriptions': 0,
        'speaker_counts': {},
        'unique_speakers': 0
    })

    return mock_service


@pytest.fixture
def create_transcription_frame():
    """
    Factory fixture to create TranscriptionFrame objects with speaker diarization.

    Returns:
        Callable that creates TranscriptionFrame with speaker ID
    """
    def _create(text: str, speaker_id: int = 0, is_final: bool = True) -> TranscriptionFrame:
        """
        Create a mock TranscriptionFrame.

        Args:
            text: Transcribed text
            speaker_id: Speaker identifier (0 or 1)
            is_final: Whether this is a final transcription
        """
        frame = TranscriptionFrame(text=text, user_id=str(speaker_id), timestamp=time.time())
        # Add speaker attribute that Deepgram would provide
        frame.speaker = speaker_id
        frame.is_final = is_final
        return frame

    return _create


# =====================================================================
# Mock LLM Service (Anthropic Claude)
# =====================================================================

@pytest.fixture
def mock_llm_service():
    """
    Mock Anthropic Claude LLM service for intervention generation.

    Returns a mock that simulates LLM intervention responses.
    """
    mock_service = AsyncMock()

    # Default intervention responses
    mock_responses = {
        'high_tension': "Let's take a breath. What outcome would satisfy both of you?",
        'speaker_imbalance': "I notice one person has been speaking more. Let's ensure both voices are heard.",
        'protocol_violation_past': "PROTOCOL VIOLATION (Rule 3): Future Focused. Please restate using forward-looking impact.",
        'protocol_warning_opinion': "PROTOCOL WARNING (Rule 2): Data Over Opinion. Can you support that with metrics?",
        'default': "Let's stay focused on reaching a decision."
    }

    async def mock_generate(context: Dict) -> str:
        """Generate mock intervention based on context."""
        reason = context.get('reason', '').lower()

        if 'past' in reason or 'history' in reason:
            return mock_responses['protocol_violation_past']
        elif 'opinion' in reason or 'data' in reason:
            return mock_responses['protocol_warning_opinion']
        elif context.get('tension_score', 0) > 0.7:
            return mock_responses['high_tension']
        elif 'imbalance' in reason or 'dominating' in reason:
            return mock_responses['speaker_imbalance']
        else:
            return mock_responses['default']

    mock_service.generate_intervention = mock_generate
    mock_service.shutdown = Mock()

    return mock_service


@pytest.fixture
def mock_anthropic_api():
    """
    Mock the Anthropic API client for direct API calls.
    """
    with patch('anthropic.AsyncAnthropic') as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Create mock response
        async def mock_create(**kwargs):
            mock_response = Mock()
            mock_response.content = [
                Mock(text="Let's take a moment to ensure both perspectives are heard.")
            ]
            return mock_response

        mock_client.messages.create = mock_create

        yield mock_client


# =====================================================================
# Mock TTS Service (ElevenLabs)
# =====================================================================

@pytest.fixture
def mock_tts_service():
    """
    Mock ElevenLabs TTS service.

    Simulates text-to-speech conversion without actual API calls.
    """
    mock_service = AsyncMock()
    mock_service.process_frame = AsyncMock()

    # Track TTS calls for verification
    mock_service.tts_calls = []

    async def mock_process(frame, direction):
        """Track TTS processing calls."""
        if isinstance(frame, TextFrame):
            mock_service.tts_calls.append({
                'text': frame.text,
                'timestamp': time.time()
            })

    mock_service.process_frame = mock_process

    return mock_service


# =====================================================================
# Mock Daily.co Transport
# =====================================================================

@pytest.fixture
def mock_daily_transport():
    """
    Mock Daily.co WebRTC transport.

    Simulates the real-time audio/video connection.
    """
    mock_transport = AsyncMock()

    # Connection state
    mock_transport.connected = True
    mock_transport.participant_count = 2

    # Audio/video state
    mock_transport.audio_enabled = True
    mock_transport.video_enabled = False

    # Methods
    mock_transport.start = AsyncMock()
    mock_transport.stop = AsyncMock()
    mock_transport.send_audio = AsyncMock()
    mock_transport.send_frame = AsyncMock()

    # Track sent frames
    mock_transport.sent_frames = []

    async def mock_send(frame):
        """Track frames sent through transport."""
        mock_transport.sent_frames.append(frame)

    mock_transport.send_frame = mock_send

    return mock_transport


# =====================================================================
# Test Data Generators
# =====================================================================

@pytest.fixture
def sample_utterances():
    """
    Generate sample utterances for testing different scenarios.
    """
    def _generate(scenario: str) -> List[Dict[str, Any]]:
        """
        Generate utterances for a specific test scenario.

        Args:
            scenario: Type of scenario (calm, high_tension, imbalanced, etc.)
        """
        if scenario == 'calm':
            return [
                {'speaker': 0, 'text': 'I think we should focus on the Q1 metrics.', 'duration': 2.5},
                {'speaker': 1, 'text': 'Good point. The data shows 15% growth.', 'duration': 2.0},
                {'speaker': 0, 'text': 'That aligns with our projections.', 'duration': 1.5},
                {'speaker': 1, 'text': "Let's plan the next steps.", 'duration': 1.8},
            ]

        elif scenario == 'high_tension':
            return [
                {'speaker': 0, 'text': "You're always wrong about the marketing strategy!", 'duration': 2.0},
                {'speaker': 1, 'text': "That's your fault for not listening to data!", 'duration': 2.2},
                {'speaker': 0, 'text': "This is ridiculous! You never accept my ideas!", 'duration': 2.5},
                {'speaker': 1, 'text': "Your ideas are always impossible to implement!", 'duration': 2.3},
            ]

        elif scenario == 'imbalanced':
            # Speaker 0 dominates with 80%+ talk time
            return [
                {'speaker': 0, 'text': 'I analyzed all the quarterly data and found several key insights about our customer acquisition funnel.', 'duration': 5.0},
                {'speaker': 0, 'text': 'The conversion rate dropped by 12% but engagement metrics are up 25% which suggests a quality issue.', 'duration': 6.0},
                {'speaker': 0, 'text': 'We need to completely redesign the onboarding flow based on user feedback from the last survey.', 'duration': 5.5},
                {'speaker': 1, 'text': 'Okay.', 'duration': 0.5},
                {'speaker': 0, 'text': 'Furthermore, the retention data indicates we should prioritize feature X over feature Y.', 'duration': 5.0},
                {'speaker': 0, 'text': 'I have more detailed analysis in the spreadsheet if you want to review it later.', 'duration': 4.0},
            ]

        elif scenario == 'past_reference':
            return [
                {'speaker': 0, 'text': 'Last year we agreed to launch in Q2.', 'duration': 2.5},
                {'speaker': 1, 'text': 'But you always change the timeline.', 'duration': 2.0},
                {'speaker': 0, 'text': 'Remember when we failed before?', 'duration': 1.8},
            ]

        elif scenario == 'opinion_based':
            return [
                {'speaker': 0, 'text': 'I feel like we should pivot immediately.', 'duration': 2.2},
                {'speaker': 1, 'text': 'I think that seems risky.', 'duration': 1.8},
                {'speaker': 0, 'text': 'In my opinion, the market is shifting.', 'duration': 2.0},
            ]

        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    return _generate


@pytest.fixture
def mock_founder_names():
    """Provide default founder names for testing."""
    return ["Founder A", "Founder B"]


# =====================================================================
# Async Test Helpers
# =====================================================================

@pytest.fixture
def event_loop():
    """
    Create an instance of the default event loop for the test session.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def async_timeout():
    """
    Provide a timeout context manager for async tests.
    """
    async def _timeout(seconds: float):
        """Wait with timeout."""
        await asyncio.sleep(seconds)

    return _timeout


# =====================================================================
# Memory/State Storage for Tests
# =====================================================================

@pytest.fixture
def memory_store():
    """
    In-memory storage for testing memory coordination between components.

    Simulates the memory namespace used for agent coordination.
    """
    store = {}

    def set_value(namespace: str, key: str, value: Any):
        """Store a value."""
        full_key = f"{namespace}:{key}"
        store[full_key] = value

    def get_value(namespace: str, key: str) -> Any:
        """Retrieve a value."""
        full_key = f"{namespace}:{key}"
        return store.get(full_key)

    def clear():
        """Clear all stored values."""
        store.clear()

    return {
        'set': set_value,
        'get': get_value,
        'clear': clear,
        'data': store
    }


# =====================================================================
# Configuration Fixtures
# =====================================================================

@pytest.fixture
def test_config():
    """
    Provide test configuration for processors and services.
    """
    return {
        'tension_threshold': 0.7,
        'cooldown_seconds': 60,
        'buffer_size': 50,
        'speaker_imbalance_threshold': 0.8,
        'speaker_imbalance_min_duration': 300,
    }
