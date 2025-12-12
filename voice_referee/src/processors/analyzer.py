"""
Conversation Analysis Engine for Voice Referee System

This module provides real-time analysis of conversation dynamics including:
- Tension score calculation
- Sentiment analysis
- Interruption detection
- Argument repetition detection
- Speaker balance analysis
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import re
from collections import Counter
from datetime import datetime


@dataclass
class Utterance:
    """Represents a single utterance in the conversation"""
    speaker: str
    text: str
    timestamp: float
    duration: float
    is_interruption: bool = False


class ConversationAnalyzer:
    """
    Analyzes conversation dynamics to detect tension, imbalance, and repetition.

    The analyzer computes a tension score based on multiple factors:
    - Sentiment negativity (30%)
    - Interruption rate (30%)
    - Speaker imbalance (20%)
    - Argument repetition (20%)
    """

    # Tension indicators - keywords that signal conflict
    TENSION_KEYWORDS = [
        "never", "always", "wrong", "fault", "stupid",
        "ridiculous", "unfair", "impossible", "terrible",
        "awful", "worst", "hate", "can't", "won't"
    ]

    # Negative sentiment indicators
    NEGATIVE_KEYWORDS = [
        "bad", "worse", "fail", "failed", "problem", "issue",
        "concern", "worried", "angry", "frustrated", "annoyed"
    ]

    # Positive sentiment indicators (for balance)
    POSITIVE_KEYWORDS = [
        "good", "great", "excellent", "agree", "yes", "right",
        "correct", "perfect", "appreciate", "thank", "thanks"
    ]

    def __init__(self):
        """Initialize the conversation analyzer"""
        self.interruption_count = 0
        self.total_utterances = 0
        self.speaker_word_counts: Dict[str, int] = {}
        self.recent_topics: List[str] = []

    def calculate_tension_score(self, transcript: List[Utterance]) -> float:
        """
        Calculate overall tension score from 0.0 (calm) to 1.0 (high tension).

        Formula:
        Tension_Score = weighted_sum(
            sentiment_negativity * 0.3,
            interruption_rate * 0.3,
            speaker_imbalance * 0.2,
            argument_repetition * 0.2
        )

        Args:
            transcript: List of utterances to analyze

        Returns:
            float: Tension score between 0.0 and 1.0
        """
        if not transcript:
            return 0.0

        # Calculate component scores
        sentiment_score = self._calculate_sentiment_negativity(transcript)
        interruption_score = self._calculate_interruption_rate(transcript)
        imbalance_score = self._calculate_speaker_imbalance(transcript)
        repetition_score = self._calculate_argument_repetition(transcript)

        # Weighted combination
        tension_score = (
            sentiment_score * 0.3 +
            interruption_score * 0.3 +
            imbalance_score * 0.2 +
            repetition_score * 0.2
        )

        return min(1.0, max(0.0, tension_score))

    def detect_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text using keyword-based approach.

        Args:
            text: Text to analyze

        Returns:
            float: Sentiment score from -1.0 (negative) to 1.0 (positive)
        """
        if not text:
            return 0.0

        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        # Count tension and negative keywords
        tension_count = sum(1 for word in words if word in self.TENSION_KEYWORDS)
        negative_count = sum(1 for word in words if word in self.NEGATIVE_KEYWORDS)
        positive_count = sum(1 for word in words if word in self.POSITIVE_KEYWORDS)

        # Calculate raw sentiment
        total_sentiment_words = tension_count + negative_count + positive_count
        if total_sentiment_words == 0:
            return 0.0

        # Tension keywords count double
        negative_score = (tension_count * 2 + negative_count) / len(words)
        positive_score = positive_count / len(words)

        # Normalize to -1 to 1 range
        sentiment = positive_score - negative_score
        return max(-1.0, min(1.0, sentiment * 10))  # Scale for sensitivity

    def detect_argument_repetition(self, recent_utterances: List[Utterance]) -> int:
        """
        Detect if the same argument/topic is being repeated.

        Args:
            recent_utterances: Recent conversation utterances (typically last 5-10)

        Returns:
            int: Number of times similar arguments appear
        """
        if len(recent_utterances) < 2:
            return 0

        # Extract key terms from each utterance (nouns, verbs, adjectives)
        utterance_keywords = []
        for utterance in recent_utterances:
            words = re.findall(r'\b\w{4,}\b', utterance.text.lower())  # 4+ char words
            # Filter out common words
            filtered = [w for w in words if w not in {'that', 'this', 'with', 'have', 'been', 'were', 'their'}]
            utterance_keywords.append(set(filtered[:10]))  # Top 10 keywords

        # Count overlapping keyword sets
        repetition_count = 0
        for i in range(len(utterance_keywords)):
            for j in range(i + 1, len(utterance_keywords)):
                overlap = utterance_keywords[i] & utterance_keywords[j]
                # If 40% or more keywords overlap, consider it repetition
                if len(overlap) >= 0.4 * min(len(utterance_keywords[i]), len(utterance_keywords[j])):
                    repetition_count += 1

        return repetition_count

    def calculate_interruption_rate(self, transcript: List[Utterance]) -> float:
        """
        Calculate the rate of interruptions in the conversation.

        Args:
            transcript: List of utterances

        Returns:
            float: Interruption rate from 0.0 to 1.0
        """
        if not transcript:
            return 0.0

        interruption_count = sum(1 for u in transcript if u.is_interruption)
        return interruption_count / len(transcript)

    def _calculate_sentiment_negativity(self, transcript: List[Utterance]) -> float:
        """Calculate average sentiment negativity (0.0 = positive, 1.0 = negative)"""
        if not transcript:
            return 0.0

        sentiments = [self.detect_sentiment(u.text) for u in transcript]
        avg_sentiment = sum(sentiments) / len(sentiments)

        # Convert from [-1, 1] to [0, 1] where 1 is most negative
        return (1.0 - avg_sentiment) / 2.0

    def _calculate_interruption_rate(self, transcript: List[Utterance]) -> float:
        """Calculate interruption rate as a 0-1 score"""
        return self.calculate_interruption_rate(transcript)

    def _calculate_speaker_imbalance(self, transcript: List[Utterance]) -> float:
        """
        Calculate speaker imbalance (0.0 = balanced, 1.0 = one speaker dominates).

        Uses word count rather than utterance count for more accurate balance.
        """
        if not transcript:
            return 0.0

        # Count words per speaker
        speaker_words: Dict[str, int] = {}
        for utterance in transcript:
            words = len(utterance.text.split())
            speaker_words[utterance.speaker] = speaker_words.get(utterance.speaker, 0) + words

        if len(speaker_words) < 2:
            return 0.0

        # Calculate imbalance ratio
        word_counts = list(speaker_words.values())
        max_words = max(word_counts)
        min_words = min(word_counts)
        total_words = sum(word_counts)

        if total_words == 0:
            return 0.0

        # Imbalance is the deviation from equal distribution
        # Perfect balance (50/50) = 0.0, complete imbalance (100/0) = 1.0
        expected_per_speaker = total_words / len(speaker_words)
        max_deviation = max(abs(count - expected_per_speaker) for count in word_counts)

        return max_deviation / expected_per_speaker if expected_per_speaker > 0 else 0.0

    def _calculate_argument_repetition(self, transcript: List[Utterance]) -> float:
        """
        Calculate argument repetition score (0.0 = no repetition, 1.0 = high repetition).

        Analyzes the last 10 utterances for topic repetition.
        """
        # Look at recent conversation (last 10 utterances)
        recent = transcript[-10:] if len(transcript) > 10 else transcript

        if len(recent) < 3:
            return 0.0

        repetition_count = self.detect_argument_repetition(recent)

        # Normalize: 3+ repetitions = 1.0, 0 repetitions = 0.0
        return min(1.0, repetition_count / 3.0)

    def get_analysis_summary(self, transcript: List[Utterance]) -> Dict[str, float]:
        """
        Get a complete analysis summary of the conversation.

        Args:
            transcript: List of utterances to analyze

        Returns:
            dict: Analysis metrics including tension score and components
        """
        tension_score = self.calculate_tension_score(transcript)

        return {
            'tension_score': tension_score,
            'sentiment_negativity': self._calculate_sentiment_negativity(transcript),
            'interruption_rate': self._calculate_interruption_rate(transcript),
            'speaker_imbalance': self._calculate_speaker_imbalance(transcript),
            'argument_repetition': self._calculate_argument_repetition(transcript),
            'utterance_count': len(transcript),
            'timestamp': datetime.now().isoformat()
        }

    def analyze(self, state) -> Dict:
        """
        Analyze conversation state and return analysis results.

        Args:
            state: ConversationState object with transcript_buffer

        Returns:
            dict: Analysis results including tension_score and recommend_intervention
        """
        # Convert ConversationState utterances to analyzer format
        transcript = []
        for u in state.transcript_buffer:
            utterance = Utterance(
                speaker=u.speaker,
                text=u.text,
                timestamp=u.timestamp,
                duration=u.duration,
                is_interruption=False
            )
            transcript.append(utterance)

        summary = self.get_analysis_summary(transcript)
        summary['recommend_intervention'] = summary['tension_score'] > 0.7
        return summary
