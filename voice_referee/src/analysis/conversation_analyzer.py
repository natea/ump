"""
Conversation Analyzer Module

Analyzes conversation state to detect tension, imbalance, and other metrics.
"""

from dataclasses import dataclass
from typing import List, Optional
import logging

from processors.conversation_state import ConversationState, Utterance

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """
    Result of conversation analysis.

    Attributes:
        tension_score: Overall tension level (0.0 to 1.0)
        balance_score: Speaking time balance (0.0 = balanced, 1.0 = imbalanced)
        interruption_rate: Interruptions per minute
        dominant_speaker: Speaker with most speaking time (or None)
        detected_patterns: List of detected conversation patterns
        requires_intervention: Whether intervention is recommended
    """
    tension_score: float
    balance_score: float
    interruption_rate: float
    dominant_speaker: Optional[str]
    detected_patterns: List[str]
    requires_intervention: bool


class ConversationAnalyzer:
    """
    Analyzes conversation metrics to detect when intervention is needed.

    Uses conversation state to calculate:
    - Tension levels (based on interruptions, speaking patterns)
    - Speaking time balance
    - Conversation health indicators
    """

    def __init__(self, tension_threshold: float = 0.7):
        """
        Initialize the analyzer.

        Args:
            tension_threshold: Threshold above which intervention is recommended (0.0-1.0)
        """
        if not 0.0 <= tension_threshold <= 1.0:
            raise ValueError("tension_threshold must be between 0.0 and 1.0")

        self._tension_threshold = tension_threshold
        logger.info(f"ConversationAnalyzer initialized with tension_threshold={tension_threshold}")

    def analyze(self, state: ConversationState) -> AnalysisResult:
        """
        Analyze the current conversation state.

        Args:
            state: Current conversation state

        Returns:
            AnalysisResult with analysis metrics
        """
        stats = state.get_stats()

        # Calculate tension score based on multiple factors
        tension_score = self._calculate_tension(state, stats)

        # Get balance score from state
        balance_score = stats['balance_score']

        # Calculate interruption rate (interruptions per minute)
        session_duration_minutes = stats['session_duration'] / 60.0
        interruption_rate = (
            stats['interruption_count'] / session_duration_minutes
            if session_duration_minutes > 0 else 0.0
        )

        # Determine dominant speaker
        dominant_speaker = self._find_dominant_speaker(stats['speaker_stats'])

        # Detect conversation patterns
        detected_patterns = self._detect_patterns(state, stats)

        # Determine if intervention is required
        requires_intervention = tension_score >= self._tension_threshold

        result = AnalysisResult(
            tension_score=tension_score,
            balance_score=balance_score,
            interruption_rate=interruption_rate,
            dominant_speaker=dominant_speaker,
            detected_patterns=detected_patterns,
            requires_intervention=requires_intervention
        )

        logger.info(
            f"📊 Analysis: tension={tension_score:.2f}, balance={balance_score:.2f}, "
            f"interruption_rate={interruption_rate:.2f}, patterns={detected_patterns}, intervention={requires_intervention}"
        )

        return result

    def _calculate_tension(self, state: ConversationState, stats: dict) -> float:
        """
        Calculate overall tension score from multiple factors.

        Args:
            state: Conversation state
            stats: Statistics dictionary from state

        Returns:
            Tension score from 0.0 to 1.0
        """
        # Factor 1: Interruption frequency (normalized to 0-1)
        # Assume 10+ interruptions per minute is maximum tension
        session_duration_minutes = stats['session_duration'] / 60.0
        interruption_rate = (
            stats['interruption_count'] / session_duration_minutes
            if session_duration_minutes > 0 else 0.0
        )
        interruption_factor = min(interruption_rate / 10.0, 1.0)

        # Factor 2: Speaking imbalance
        balance_factor = stats['balance_score']

        # Factor 3: Average negative sentiment (if available)
        sentiment_factor = 0.0
        speaker_stats = stats['speaker_stats']
        if speaker_stats:
            avg_sentiment = sum(s['avg_sentiment'] for s in speaker_stats.values()) / len(speaker_stats)
            # Convert negative sentiment to tension (0 sentiment = 0.5 tension, -1 = 1.0 tension)
            sentiment_factor = max(0.0, 0.5 - avg_sentiment / 2.0)

        # Weighted combination of factors
        tension_score = (
            0.4 * interruption_factor +
            0.3 * balance_factor +
            0.3 * sentiment_factor
        )

        return min(tension_score, 1.0)

    def _find_dominant_speaker(self, speaker_stats: dict) -> Optional[str]:
        """
        Find the speaker with the most speaking time.

        Args:
            speaker_stats: Dictionary of speaker statistics

        Returns:
            Name of dominant speaker or None if no speakers
        """
        if not speaker_stats:
            return None

        # Find speaker with maximum speaking percentage
        dominant = max(
            speaker_stats.items(),
            key=lambda x: x[1]['speaking_percentage']
        )

        # Only return if they speak significantly more (>60%)
        if dominant[1]['speaking_percentage'] > 60.0:
            return dominant[0]

        return None

    def _detect_patterns(self, state: ConversationState, stats: dict) -> List[str]:
        """
        Detect conversation patterns that may need intervention.

        Args:
            state: Conversation state
            stats: Statistics dictionary

        Returns:
            List of detected pattern descriptions
        """
        patterns = []

        # Pattern: One person dominating
        dominant = self._find_dominant_speaker(stats['speaker_stats'])
        if dominant:
            patterns.append(f"{dominant} is dominating the conversation")

        # Pattern: High interruption rate
        session_duration_minutes = stats['session_duration'] / 60.0
        if session_duration_minutes > 0:
            interruption_rate = stats['interruption_count'] / session_duration_minutes
            if interruption_rate > 5.0:
                patterns.append("High interruption frequency detected")

        # Pattern: Extreme imbalance
        if stats['balance_score'] > 0.8:
            patterns.append("Severe speaking time imbalance")

        # Pattern: Recent rapid-fire exchanges
        recent_utterances = state.get_recent_transcript(n=5)
        if len(recent_utterances) >= 5:
            # Check if last 5 utterances happened within 30 seconds
            time_span = recent_utterances[-1].timestamp - recent_utterances[0].timestamp
            if time_span < 30.0:
                patterns.append("Rapid-fire exchange detected")

        return patterns

    def set_tension_threshold(self, threshold: float):
        """
        Update the tension threshold.

        Args:
            threshold: New threshold value (0.0-1.0)

        Raises:
            ValueError: If threshold is out of range
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("tension_threshold must be between 0.0 and 1.0")

        logger.info(f"Tension threshold updated: {self._tension_threshold} -> {threshold}")
        self._tension_threshold = threshold

    def __repr__(self) -> str:
        """String representation."""
        return f"ConversationAnalyzer(tension_threshold={self._tension_threshold})"
