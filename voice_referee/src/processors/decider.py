"""
Intervention Decision Engine for Voice Referee System

This module implements the decision logic for when and how to intervene
in conversations based on ump.ai protocols and conversation analysis.

Protocols:
1. No Interruptions - Allow complete thoughts
2. Data Over Opinion - Cite specific metrics
3. Future Focused - No dredging past issues
4. Binary Outcome - Commit to decision by session end
"""

from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import re


@dataclass
class ConversationState:
    """Current state of the conversation"""
    duration_seconds: float
    tension_score: float
    interruption_rate: float
    speaker_imbalance: float
    argument_repetition_count: int
    last_intervention_time: Optional[datetime]
    total_interventions: int
    current_speakers: List[str]


class InterventionDecider:
    """
    Decides when and how to intervene in conversations based on:
    - Tension levels
    - Protocol violations
    - Speaker balance
    - Argument repetition

    Implements ump.ai protocols for productive conversations.
    """

    # Intervention thresholds
    TENSION_THRESHOLD = 0.7
    SPEAKER_IMBALANCE_THRESHOLD = 0.8
    SPEAKER_IMBALANCE_MIN_DURATION = 300  # 5 minutes
    ARGUMENT_REPETITION_THRESHOLD = 3

    # Cooldown period between interventions (seconds)
    INTERVENTION_COOLDOWN = 60  # 1 minute

    # Protocol 2: Data Over Opinion - Opinion phrases
    OPINION_PHRASES = [
        r'\bi feel\b',
        r'\bi think\b',
        r'\bi believe\b',
        r'\bin my opinion\b',
        r'\bseems like\b',
        r'\bprobably\b',
        r'\bmaybe\b'
    ]

    # Protocol 2: Data indicators (good)
    DATA_INDICATORS = [
        r'\d+%',  # Percentages
        r'\$\d+',  # Dollar amounts
        r'\b\d+\s*(users|customers|people|times|days|hours)\b',  # Metrics
        r'\bdata shows\b',
        r'\bmetrics indicate\b',
        r'\baccording to\b',
        r'\bmeasured\b',
        r'\bstatistics\b'
    ]

    # Protocol 3: Future Focused - Past references
    PAST_REFERENCES = [
        r'\blast (year|month|week|time)\b',
        r'\bback (then|when|in)\b',
        r'\bpreviously\b',
        r'\bbefore\b',
        r'\bused to\b',
        r'\bin the past\b',
        r'\byou always\b',
        r'\byou never\b'
    ]

    # Intervention types
    INTERVENTION_SYSTEM = "SYSTEM INTERVENTION"
    INTERVENTION_PROTOCOL_VIOLATION = "PROTOCOL VIOLATION"
    INTERVENTION_PROTOCOL_WARNING = "PROTOCOL WARNING"

    def __init__(self):
        """Initialize the intervention decider"""
        self.intervention_history: List[Dict] = []

    def should_intervene(
        self,
        state: ConversationState,
        analysis: Dict,
        recent_utterances: Optional[List[str]] = None
    ) -> Tuple[bool, str, str]:
        """
        Determine if intervention is needed and what type.

        Args:
            state: Current conversation state
            analysis: Analysis results from ConversationAnalyzer
            recent_utterances: Recent utterance texts for protocol checking

        Returns:
            Tuple[bool, str, str]: (should_intervene, reason, intervention_type)
        """
        # Check cooldown first
        if not self.check_cooldown(state.last_intervention_time):
            return False, "", ""

        # Check Protocol 3: Future Focused (highest priority - immediate violation)
        if recent_utterances:
            past_violation = self._check_past_references(recent_utterances)
            if past_violation:
                return True, past_violation, self.INTERVENTION_PROTOCOL_VIOLATION

        # Check Protocol 2: Data Over Opinion
        if recent_utterances:
            opinion_warning = self._check_opinion_vs_data(recent_utterances)
            if opinion_warning:
                return True, opinion_warning, self.INTERVENTION_PROTOCOL_WARNING

        # Check tension score (System Intervention)
        if state.tension_score > self.TENSION_THRESHOLD:
            reason = (
                f"High tension detected (score: {state.tension_score:.2f}). "
                f"Let's take a breath and reset."
            )
            return True, reason, self.INTERVENTION_SYSTEM

        # Check argument repetition (System Intervention)
        if state.argument_repetition_count >= self.ARGUMENT_REPETITION_THRESHOLD:
            reason = (
                f"The same points are being repeated ({state.argument_repetition_count} times). "
                f"Let's move forward with a decision."
            )
            return True, reason, self.INTERVENTION_SYSTEM

        # Check speaker imbalance (System Intervention)
        if (state.speaker_imbalance > self.SPEAKER_IMBALANCE_THRESHOLD and
            state.duration_seconds > self.SPEAKER_IMBALANCE_MIN_DURATION):
            reason = (
                f"One speaker has been dominating the conversation "
                f"(imbalance: {state.speaker_imbalance:.2f}). "
                f"Let's ensure everyone's voice is heard."
            )
            return True, reason, self.INTERVENTION_SYSTEM

        # No intervention needed
        return False, "", ""

    def check_cooldown(self, last_intervention_time: Optional[datetime]) -> bool:
        """
        Check if enough time has passed since last intervention.

        Args:
            last_intervention_time: Timestamp of last intervention

        Returns:
            bool: True if cooldown period has elapsed
        """
        if last_intervention_time is None:
            return True

        elapsed = (datetime.now() - last_intervention_time).total_seconds()
        return elapsed >= self.INTERVENTION_COOLDOWN

    def get_intervention_context(
        self,
        state: ConversationState,
        analysis: Dict
    ) -> Dict:
        """
        Get contextual information about the intervention.

        Args:
            state: Current conversation state
            analysis: Analysis results

        Returns:
            dict: Context information for the intervention
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': state.duration_seconds,
            'tension_score': state.tension_score,
            'interruption_rate': state.interruption_rate,
            'speaker_imbalance': state.speaker_imbalance,
            'argument_repetition_count': state.argument_repetition_count,
            'total_interventions': state.total_interventions,
            'speakers': state.current_speakers,
            'analysis_summary': analysis
        }

    def _check_past_references(self, recent_utterances: List[str]) -> Optional[str]:
        """
        Check for Protocol 3 violations: references to past issues.

        Args:
            recent_utterances: Recent conversation text

        Returns:
            Optional[str]: Violation message if detected, None otherwise
        """
        # Check last 3 utterances for past references
        recent_text = ' '.join(recent_utterances[-3:]).lower()

        for pattern in self.PAST_REFERENCES:
            if re.search(pattern, recent_text, re.IGNORECASE):
                return (
                    "Protocol 3 Violation: Future Focused. "
                    "Let's focus on what we can do moving forward, "
                    "not what happened in the past."
                )

        return None

    def _check_opinion_vs_data(self, recent_utterances: List[str]) -> Optional[str]:
        """
        Check for Protocol 2: Data Over Opinion.

        Args:
            recent_utterances: Recent conversation text

        Returns:
            Optional[str]: Warning message if opinion without data detected
        """
        # Check last 2 utterances
        recent_text = ' '.join(recent_utterances[-2:]).lower()

        # Check if opinion phrases are used
        has_opinion = any(
            re.search(pattern, recent_text, re.IGNORECASE)
            for pattern in self.OPINION_PHRASES
        )

        if not has_opinion:
            return None

        # Check if data/metrics are also provided
        has_data = any(
            re.search(pattern, recent_text, re.IGNORECASE)
            for pattern in self.DATA_INDICATORS
        )

        if has_opinion and not has_data:
            return (
                "Protocol 2 Reminder: Data Over Opinion. "
                "Can you support that with specific metrics or data?"
            )

        return None

    def record_intervention(
        self,
        intervention_type: str,
        reason: str,
        context: Dict
    ) -> None:
        """
        Record an intervention in the history.

        Args:
            intervention_type: Type of intervention
            reason: Reason for intervention
            context: Contextual information
        """
        self.intervention_history.append({
            'timestamp': datetime.now().isoformat(),
            'type': intervention_type,
            'reason': reason,
            'context': context
        })

    def get_intervention_stats(self) -> Dict:
        """
        Get statistics about interventions.

        Returns:
            dict: Intervention statistics
        """
        if not self.intervention_history:
            return {
                'total_interventions': 0,
                'by_type': {},
                'average_interval_seconds': 0
            }

        # Count by type
        by_type = {}
        for intervention in self.intervention_history:
            itype = intervention['type']
            by_type[itype] = by_type.get(itype, 0) + 1

        # Calculate average interval
        if len(self.intervention_history) > 1:
            timestamps = [
                datetime.fromisoformat(i['timestamp'])
                for i in self.intervention_history
            ]
            intervals = [
                (timestamps[i+1] - timestamps[i]).total_seconds()
                for i in range(len(timestamps) - 1)
            ]
            avg_interval = sum(intervals) / len(intervals)
        else:
            avg_interval = 0

        return {
            'total_interventions': len(self.intervention_history),
            'by_type': by_type,
            'average_interval_seconds': avg_interval,
            'last_intervention': self.intervention_history[-1]['timestamp']
        }
