"""
Unit tests for ConversationState module

Tests transcript buffer management, speaker statistics calculation,
balance calculation, and utterance tracking.
"""

import pytest
import time
from processors.conversation_state import (
    ConversationState,
    Utterance,
    SpeakerStats
)


class TestUtterance:
    """Test suite for Utterance dataclass"""

    def test_utterance_creation(self):
        """Test creating a valid utterance"""
        utterance = Utterance(
            text="Hello world",
            speaker="Founder A",
            timestamp=time.time(),
            duration=2.5,
            word_count=2
        )

        assert utterance.text == "Hello world"
        assert utterance.speaker == "Founder A"
        assert utterance.word_count == 2
        assert utterance.duration == 2.5

    def test_utterance_negative_word_count_raises_error(self):
        """Test that negative word_count raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            Utterance(
                text="Test",
                speaker="Founder A",
                timestamp=time.time(),
                duration=1.0,
                word_count=-5
            )

        assert "word_count cannot be negative" in str(exc_info.value)

    def test_utterance_negative_duration_raises_error(self):
        """Test that negative duration raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            Utterance(
                text="Test",
                speaker="Founder A",
                timestamp=time.time(),
                duration=-2.0,
                word_count=1
            )

        assert "duration cannot be negative" in str(exc_info.value)


class TestSpeakerStats:
    """Test suite for SpeakerStats dataclass"""

    def test_speaker_stats_defaults(self):
        """Test SpeakerStats default values"""
        stats = SpeakerStats()

        assert stats.total_time == 0.0
        assert stats.utterance_count == 0
        assert stats.word_count == 0
        assert stats.avg_sentiment == 0.0

    def test_update_from_utterance(self):
        """Test updating stats from an utterance"""
        stats = SpeakerStats()

        stats.update_from_utterance(duration=5.0, words=10, sentiment=0.5)

        assert stats.total_time == 5.0
        assert stats.utterance_count == 1
        assert stats.word_count == 10
        assert stats.avg_sentiment == 0.5

    def test_update_from_utterance_running_average(self):
        """Test that sentiment is calculated as running average"""
        stats = SpeakerStats()

        # First utterance: sentiment 0.5
        stats.update_from_utterance(duration=2.0, words=5, sentiment=0.5)
        assert stats.avg_sentiment == 0.5

        # Second utterance: sentiment 1.0
        stats.update_from_utterance(duration=3.0, words=7, sentiment=1.0)
        assert stats.avg_sentiment == 0.75  # (0.5 + 1.0) / 2

        # Third utterance: sentiment 0.0
        stats.update_from_utterance(duration=1.0, words=3, sentiment=0.0)
        assert stats.avg_sentiment == 0.5  # (0.5 + 1.0 + 0.0) / 3

    def test_update_accumulates_totals(self):
        """Test that totals accumulate correctly"""
        stats = SpeakerStats()

        stats.update_from_utterance(duration=2.0, words=5, sentiment=0.0)
        stats.update_from_utterance(duration=3.0, words=10, sentiment=0.0)
        stats.update_from_utterance(duration=1.5, words=3, sentiment=0.0)

        assert stats.total_time == 6.5
        assert stats.utterance_count == 3
        assert stats.word_count == 18


class TestConversationState:
    """Test suite for ConversationState class"""

    def test_initialization(self):
        """Test ConversationState initialization with default buffer size"""
        state = ConversationState()

        assert len(state.get_recent_transcript(50)) == 0
        assert state.calculate_speaker_balance() == 0.0

    def test_custom_buffer_size(self):
        """Test ConversationState with custom buffer size"""
        state = ConversationState(max_buffer_size=10)

        stats = state.get_stats()
        assert stats['buffer_utilization'] == 0.0

    def test_transcript_buffer_max_50(self):
        """Test that transcript buffer respects max size of 50 (default)"""
        state = ConversationState(max_buffer_size=50)

        # Add 60 utterances
        for i in range(60):
            state.add_utterance(
                text=f"Utterance {i}",
                speaker="Founder A",
                timestamp=time.time(),
                duration=1.0
            )

        # Should only keep last 50
        recent = state.get_recent_transcript(100)
        assert len(recent) == 50
        assert recent[-1].text == "Utterance 59"
        assert recent[0].text == "Utterance 10"

    def test_add_utterance_updates_stats(self):
        """Test that adding utterance updates speaker statistics"""
        state = ConversationState()

        state.add_utterance(
            text="Hello world test",
            speaker="Founder A",
            timestamp=time.time(),
            duration=2.0,
            sentiment=0.5
        )

        stats = state.get_stats()
        speaker_stats = stats['speaker_stats']['Founder A']

        assert speaker_stats['total_time'] == 2.0
        assert speaker_stats['utterance_count'] == 1
        assert speaker_stats['word_count'] == 3  # "Hello world test"
        assert speaker_stats['avg_sentiment'] == 0.5

    def test_add_utterance_returns_utterance_object(self):
        """Test that add_utterance returns created Utterance"""
        state = ConversationState()

        utterance = state.add_utterance(
            text="Test utterance",
            speaker="Founder A",
            timestamp=123.45,
            duration=1.5
        )

        assert isinstance(utterance, Utterance)
        assert utterance.text == "Test utterance"
        assert utterance.speaker == "Founder A"
        assert utterance.timestamp == 123.45
        assert utterance.duration == 1.5
        assert utterance.word_count == 2

    def test_get_recent_transcript(self):
        """Test retrieving recent transcript subset"""
        state = ConversationState()

        # Add 10 utterances
        for i in range(10):
            state.add_utterance(
                text=f"Message {i}",
                speaker="Founder A",
                timestamp=time.time(),
                duration=1.0
            )

        # Get last 5
        recent = state.get_recent_transcript(5)

        assert len(recent) == 5
        assert recent[0].text == "Message 5"
        assert recent[-1].text == "Message 9"

    def test_get_recent_transcript_with_zero_returns_empty(self):
        """Test get_recent_transcript with n=0 returns empty list"""
        state = ConversationState()
        state.add_utterance("Test", "Founder A", time.time(), 1.0)

        recent = state.get_recent_transcript(0)

        assert recent == []

    def test_get_recent_transcript_negative_returns_empty(self):
        """Test get_recent_transcript with negative n returns empty list"""
        state = ConversationState()
        state.add_utterance("Test", "Founder A", time.time(), 1.0)

        recent = state.get_recent_transcript(-5)

        assert recent == []

    def test_balance_calculation_equal(self):
        """Test speaker balance calculation with equal speaking time"""
        state = ConversationState()

        # Founder A speaks for 10 seconds
        for _ in range(5):
            state.add_utterance(
                text="Hello world",
                speaker="Founder A",
                timestamp=time.time(),
                duration=2.0
            )

        # Founder B speaks for 10 seconds
        for _ in range(5):
            state.add_utterance(
                text="Hello world",
                speaker="Founder B",
                timestamp=time.time(),
                duration=2.0
            )

        balance = state.calculate_speaker_balance()

        # Perfect balance should be 0.0
        assert balance == 0.0

    def test_balance_calculation_imbalanced(self):
        """Test speaker balance calculation with imbalanced speaking time"""
        state = ConversationState()

        # Founder A speaks for 18 seconds
        for _ in range(9):
            state.add_utterance(
                text="Hello world",
                speaker="Founder A",
                timestamp=time.time(),
                duration=2.0
            )

        # Founder B speaks for 2 seconds
        state.add_utterance(
            text="Hi",
            speaker="Founder B",
            timestamp=time.time(),
            duration=2.0
        )

        balance = state.calculate_speaker_balance()

        # Should be significantly imbalanced (close to 1.0)
        assert balance > 0.5  # High imbalance

    def test_balance_single_speaker_returns_zero(self):
        """Test balance calculation with only one speaker returns 0.0"""
        state = ConversationState()

        state.add_utterance("Test", "Founder A", time.time(), 1.0)
        state.add_utterance("Test 2", "Founder A", time.time(), 1.0)

        balance = state.calculate_speaker_balance()

        assert balance == 0.0

    def test_balance_no_speakers_returns_zero(self):
        """Test balance calculation with no speakers returns 0.0"""
        state = ConversationState()

        balance = state.calculate_speaker_balance()

        assert balance == 0.0

    def test_track_interruption(self):
        """Test interruption tracking"""
        state = ConversationState()

        state.track_interruption("Founder A")
        state.track_interruption("Founder B")
        state.track_interruption("Founder A")

        stats = state.get_stats()
        assert stats['interruption_count'] == 3

    def test_record_intervention(self):
        """Test intervention recording"""
        state = ConversationState()

        # Initially no intervention
        stats = state.get_stats()
        assert stats['last_intervention_time'] is None
        assert stats['time_since_last_intervention'] is None

        # Record intervention
        state.record_intervention()

        # Should now have intervention timestamp
        stats = state.get_stats()
        assert stats['last_intervention_time'] is not None
        assert stats['time_since_last_intervention'] is not None
        assert stats['time_since_last_intervention'] < 1.0  # Just recorded

    def test_get_stats_comprehensive(self):
        """Test get_stats returns comprehensive statistics"""
        state = ConversationState()

        state.add_utterance("Hello", "Founder A", time.time(), 1.0, sentiment=0.5)
        state.add_utterance("Hi there", "Founder B", time.time(), 1.5, sentiment=-0.2)
        state.track_interruption("Founder A")

        stats = state.get_stats()

        assert 'session_duration' in stats
        assert 'total_utterances' in stats
        assert 'speaker_stats' in stats
        assert 'interruption_count' in stats
        assert 'balance_score' in stats
        assert 'buffer_utilization' in stats

        assert stats['total_utterances'] == 2
        assert stats['interruption_count'] == 1
        assert 'Founder A' in stats['speaker_stats']
        assert 'Founder B' in stats['speaker_stats']

    def test_get_stats_speaking_percentage(self):
        """Test that get_stats calculates speaking percentage correctly"""
        state = ConversationState()

        # Add utterances with known durations
        state.add_utterance("Test", "Founder A", time.time(), 5.0)
        state.add_utterance("Test", "Founder B", time.time(), 5.0)

        # Wait a moment to ensure session duration > 0
        time.sleep(0.1)

        stats = state.get_stats()

        # Both speakers should have speaking percentage
        assert 'speaking_percentage' in stats['speaker_stats']['Founder A']
        assert 'speaking_percentage' in stats['speaker_stats']['Founder B']

    def test_reset_clears_state(self):
        """Test that reset() clears all conversation state"""
        state = ConversationState()

        # Add data
        state.add_utterance("Test", "Founder A", time.time(), 1.0)
        state.track_interruption("Founder A")
        state.record_intervention()

        # Reset
        state.reset()

        # Verify everything is cleared
        stats = state.get_stats()
        assert stats['total_utterances'] == 0
        assert stats['interruption_count'] == 0
        assert stats['last_intervention_time'] is None
        assert len(stats['speaker_stats']) == 0

    def test_repr_shows_state(self):
        """Test __repr__ provides meaningful representation"""
        state = ConversationState(max_buffer_size=50)
        state.add_utterance("Test", "Founder A", time.time(), 1.0)
        state.track_interruption("Founder A")

        repr_str = repr(state)

        assert "ConversationState" in repr_str
        assert "buffer_size=1/50" in repr_str
        assert "speakers=1" in repr_str
        assert "interruptions=1" in repr_str

    def test_buffer_utilization_calculation(self):
        """Test buffer_utilization in stats is calculated correctly"""
        state = ConversationState(max_buffer_size=10)

        # Add 5 utterances (50% utilization)
        for i in range(5):
            state.add_utterance(f"Test {i}", "Founder A", time.time(), 1.0)

        stats = state.get_stats()
        assert stats['buffer_utilization'] == 0.5

    def test_word_count_calculation(self):
        """Test that word count is calculated correctly from text"""
        state = ConversationState()

        utterance = state.add_utterance(
            text="This is a test with five words",
            speaker="Founder A",
            timestamp=time.time(),
            duration=2.0
        )

        assert utterance.word_count == 7  # Actually 7 words

        stats = state.get_stats()
        assert stats['speaker_stats']['Founder A']['word_count'] == 7

    def test_multiple_speakers_tracked_separately(self):
        """Test that multiple speakers are tracked separately"""
        state = ConversationState()

        state.add_utterance("First", "Founder A", time.time(), 1.0, 0.5)
        state.add_utterance("Second", "Founder B", time.time(), 2.0, -0.3)
        state.add_utterance("Third", "Founder C", time.time(), 1.5, 0.0)

        stats = state.get_stats()

        assert len(stats['speaker_stats']) == 3
        assert stats['speaker_stats']['Founder A']['utterance_count'] == 1
        assert stats['speaker_stats']['Founder B']['utterance_count'] == 1
        assert stats['speaker_stats']['Founder C']['utterance_count'] == 1
