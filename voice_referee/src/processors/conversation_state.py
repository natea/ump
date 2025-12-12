"""
Conversation State Module

Manages the state of the conversation including transcript buffer,
speaker statistics, and session metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import deque
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class Utterance:
    """
    Represents a single utterance in the conversation.

    Attributes:
        text: The transcribed text
        speaker: Speaker identifier (e.g., "Founder A")
        timestamp: Unix timestamp when utterance occurred
        duration: Duration of utterance in seconds
        word_count: Number of words in the utterance
    """
    text: str
    speaker: str
    timestamp: float
    duration: float
    word_count: int

    def __post_init__(self):
        """Validate utterance data after initialization."""
        if self.word_count < 0:
            raise ValueError("word_count cannot be negative")
        if self.duration < 0:
            raise ValueError("duration cannot be negative")


@dataclass
class SpeakerStats:
    """
    Statistics for a single speaker.

    Attributes:
        total_time: Total speaking time in seconds
        utterance_count: Number of utterances
        word_count: Total number of words spoken
        avg_sentiment: Average sentiment score (-1.0 to 1.0)
    """
    total_time: float = 0.0
    utterance_count: int = 0
    word_count: int = 0
    avg_sentiment: float = 0.0

    def update_from_utterance(self, duration: float, words: int, sentiment: float = 0.0):
        """
        Update stats from a new utterance.

        Args:
            duration: Duration of the utterance in seconds
            words: Number of words in the utterance
            sentiment: Sentiment score for the utterance (-1.0 to 1.0)
        """
        # Update running average sentiment
        total_sentiment = self.avg_sentiment * self.utterance_count + sentiment
        self.utterance_count += 1
        self.avg_sentiment = total_sentiment / self.utterance_count

        # Update totals
        self.total_time += duration
        self.word_count += words


class ConversationState:
    """
    Manages the state of the conversation.

    Tracks transcript buffer, per-speaker statistics, interruptions,
    and session timing.

    Attributes:
        _transcript_buffer: Circular buffer of recent utterances (max 50)
        _per_speaker_stats: Statistics per speaker
        _interruption_count: Total number of interruptions detected
        _session_start_time: Unix timestamp of session start
        _last_intervention_time: Unix timestamp of last AI intervention
    """

    def __init__(self, max_buffer_size: int = 50):
        """
        Initialize conversation state.

        Args:
            max_buffer_size: Maximum number of utterances to keep in buffer
        """
        self._transcript_buffer: deque = deque(maxlen=max_buffer_size)
        self._per_speaker_stats: Dict[str, SpeakerStats] = {}
        self._interruption_count: int = 0
        self._session_start_time: float = time.time()
        self._last_intervention_time: Optional[float] = None
        self._max_buffer_size = max_buffer_size

        logger.info(f"ConversationState initialized with buffer size {max_buffer_size}")

    def add_utterance(
        self,
        text: str,
        speaker: str,
        timestamp: float,
        duration: float,
        sentiment: float = 0.0
    ) -> Utterance:
        """
        Add a new utterance to the conversation state.

        Args:
            text: The transcribed text
            speaker: Speaker identifier
            timestamp: Unix timestamp
            duration: Duration in seconds
            sentiment: Optional sentiment score (-1.0 to 1.0)

        Returns:
            The created Utterance object
        """
        # Calculate word count
        word_count = len(text.split())

        # Create utterance
        utterance = Utterance(
            text=text,
            speaker=speaker,
            timestamp=timestamp,
            duration=duration,
            word_count=word_count
        )

        # Add to buffer (automatically drops oldest if full)
        self._transcript_buffer.append(utterance)

        # Update speaker stats
        if speaker not in self._per_speaker_stats:
            self._per_speaker_stats[speaker] = SpeakerStats()

        self._per_speaker_stats[speaker].update_from_utterance(
            duration=duration,
            words=word_count,
            sentiment=sentiment
        )

        logger.debug(
            f"Added utterance: speaker={speaker}, words={word_count}, "
            f"duration={duration:.2f}s, buffer_size={len(self._transcript_buffer)}"
        )

        return utterance

    def get_recent_transcript(self, n: int = 10) -> List[Utterance]:
        """
        Get the n most recent utterances.

        Args:
            n: Number of recent utterances to retrieve

        Returns:
            List of recent utterances (most recent last)
        """
        if n <= 0:
            return []

        # Convert deque to list and slice
        buffer_list = list(self._transcript_buffer)
        return buffer_list[-n:]

    def calculate_speaker_balance(self) -> float:
        """
        Calculate speaking time balance between speakers.

        Returns:
            Balance score from 0.0 (perfectly balanced) to 1.0 (completely imbalanced).
            If fewer than 2 speakers, returns 0.0.
        """
        if len(self._per_speaker_stats) < 2:
            return 0.0

        # Get total speaking times
        speaking_times = [stats.total_time for stats in self._per_speaker_stats.values()]
        total_time = sum(speaking_times)

        if total_time == 0:
            return 0.0

        # Calculate normalized proportions
        proportions = [t / total_time for t in speaking_times]

        # Perfect balance would be 1/n for each speaker
        n_speakers = len(proportions)
        perfect_proportion = 1.0 / n_speakers

        # Calculate deviation from perfect balance
        # Sum of absolute differences from perfect proportion
        deviation = sum(abs(p - perfect_proportion) for p in proportions)

        # Normalize to 0-1 range
        # Maximum deviation is 2*(n-1)/n (one speaker talks 100%, others 0%)
        max_deviation = 2 * (n_speakers - 1) / n_speakers
        balance_score = deviation / max_deviation if max_deviation > 0 else 0.0

        return balance_score

    def track_interruption(self, speaker: str):
        """
        Record an interruption by a speaker.

        Args:
            speaker: The speaker who interrupted
        """
        self._interruption_count += 1
        logger.info(f"Interruption tracked: {speaker} (total: {self._interruption_count})")

    def record_intervention(self):
        """Record that an AI intervention occurred."""
        self._last_intervention_time = time.time()
        logger.info("AI intervention recorded")

    def get_stats(self) -> Dict:
        """
        Get comprehensive conversation statistics.

        Returns:
            Dictionary containing:
            - session_duration: Total session time in seconds
            - total_utterances: Total number of utterances
            - speaker_stats: Per-speaker statistics
            - interruption_count: Total interruptions
            - balance_score: Speaker balance (0.0 = balanced, 1.0 = imbalanced)
            - last_intervention_time: Timestamp of last intervention (or None)
            - time_since_last_intervention: Seconds since last intervention (or None)
        """
        current_time = time.time()
        session_duration = current_time - self._session_start_time

        time_since_intervention = None
        if self._last_intervention_time is not None:
            time_since_intervention = current_time - self._last_intervention_time

        stats = {
            'session_duration': session_duration,
            'total_utterances': len(self._transcript_buffer),
            'speaker_stats': {
                speaker: {
                    'total_time': stats.total_time,
                    'utterance_count': stats.utterance_count,
                    'word_count': stats.word_count,
                    'avg_sentiment': stats.avg_sentiment,
                    'speaking_percentage': (stats.total_time / session_duration * 100)
                        if session_duration > 0 else 0.0
                }
                for speaker, stats in self._per_speaker_stats.items()
            },
            'interruption_count': self._interruption_count,
            'balance_score': self.calculate_speaker_balance(),
            'last_intervention_time': self._last_intervention_time,
            'time_since_last_intervention': time_since_intervention,
            'buffer_utilization': len(self._transcript_buffer) / self._max_buffer_size
        }

        return stats

    def reset(self):
        """Reset conversation state for a new session."""
        logger.info("Resetting conversation state")
        self._transcript_buffer.clear()
        self._per_speaker_stats.clear()
        self._interruption_count = 0
        self._session_start_time = time.time()
        self._last_intervention_time = None

    def __repr__(self) -> str:
        """String representation of conversation state."""
        return (
            f"ConversationState("
            f"buffer_size={len(self._transcript_buffer)}/{self._max_buffer_size}, "
            f"speakers={len(self._per_speaker_stats)}, "
            f"interruptions={self._interruption_count})"
        )

    # Property accessors for compatibility
    @property
    def transcript_buffer(self) -> List[Utterance]:
        """Get transcript buffer as list."""
        return list(self._transcript_buffer)

    @property
    def last_intervention_time(self) -> Optional[float]:
        """Get last intervention time."""
        return self._last_intervention_time

    @property
    def interruption_count(self) -> int:
        """Get interruption count."""
        return self._interruption_count

    @property
    def session_start_time(self) -> float:
        """Get session start time."""
        return self._session_start_time

    @property
    def per_speaker_stats(self) -> Dict[str, SpeakerStats]:
        """Get per-speaker statistics."""
        return self._per_speaker_stats
