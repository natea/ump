"""
Intervention Decider Module

Decides whether and when the AI referee should intervene in the conversation.
"""

from dataclasses import dataclass
from typing import Optional
import time
import logging

from src.analysis.conversation_analyzer import AnalysisResult
from src.processors.conversation_state import ConversationState

logger = logging.getLogger(__name__)


@dataclass
class InterventionDecision:
    """
    Decision about whether to intervene.

    Attributes:
        should_intervene: Whether intervention is recommended
        reason: Human-readable reason for the decision
        suggested_prompt: Suggested intervention message/prompt
        confidence: Confidence in the decision (0.0 to 1.0)
        cooldown_active: Whether cooldown period is active
    """
    should_intervene: bool
    reason: str
    suggested_prompt: Optional[str]
    confidence: float
    cooldown_active: bool


class InterventionDecider:
    """
    Decides when the AI referee should intervene.

    Uses analysis results and conversation state to make decisions while
    respecting cooldown periods to avoid over-intervention.
    """

    def __init__(self, cooldown_seconds: int = 30, utterances_between_checkins: int = 5):
        """
        Initialize the decider.

        Args:
            cooldown_seconds: Minimum seconds between interventions
            utterances_between_checkins: Number of utterances between proactive check-ins
        """
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")

        self._cooldown_seconds = cooldown_seconds
        self._last_intervention_time: Optional[float] = None
        self._utterances_since_intervention = 0
        self._utterances_between_checkins = utterances_between_checkins
        self._consecutive_same_speaker = 0
        self._last_speaker: Optional[str] = None

        logger.info(f"InterventionDecider initialized with cooldown={cooldown_seconds}s, checkin_every={utterances_between_checkins} utterances")

    def decide(
        self,
        analysis: AnalysisResult,
        state: ConversationState
    ) -> InterventionDecision:
        """
        Decide whether to intervene based on analysis and state.

        Args:
            analysis: Analysis result from ConversationAnalyzer
            state: Current conversation state

        Returns:
            InterventionDecision with recommendation
        """
        current_time = time.time()

        # Track utterances for proactive engagement
        self._utterances_since_intervention += 1

        # Track consecutive same speaker
        recent = state.get_recent_transcript(n=1)
        if recent:
            current_speaker = recent[0].speaker
            if current_speaker == self._last_speaker:
                self._consecutive_same_speaker += 1
            else:
                self._consecutive_same_speaker = 1
            self._last_speaker = current_speaker

        # Check if we're in cooldown period
        cooldown_active = self._is_in_cooldown(current_time)

        # Don't intervene if in cooldown
        if cooldown_active:
            time_remaining = self._get_cooldown_remaining(current_time)
            return InterventionDecision(
                should_intervene=False,
                reason=f"In cooldown period ({time_remaining:.0f}s remaining)",
                suggested_prompt=None,
                confidence=0.0,
                cooldown_active=True
            )

        # PROACTIVE TRIGGERS (even if tension is low):

        # 1. Periodic check-in every N utterances
        if self._utterances_since_intervention >= self._utterances_between_checkins:
            logger.info(f"Proactive intervention: {self._utterances_since_intervention} utterances since last check-in")
            return InterventionDecision(
                should_intervene=True,
                reason=f"Periodic engagement check-in ({self._utterances_since_intervention} utterances)",
                suggested_prompt=self._build_checkin_prompt(analysis, state),
                confidence=0.7,
                cooldown_active=False
            )

        # 2. One person speaking too much consecutively (3+ in a row)
        if self._consecutive_same_speaker >= 3:
            logger.info(f"Proactive intervention: {self._last_speaker} has spoken {self._consecutive_same_speaker} times consecutively")
            return InterventionDecision(
                should_intervene=True,
                reason=f"{self._last_speaker} speaking consecutively ({self._consecutive_same_speaker}x)",
                suggested_prompt=self._build_balance_prompt(analysis, state),
                confidence=0.8,
                cooldown_active=False
            )

        # 3. Check if analysis recommends intervention (tension-based)
        if analysis.requires_intervention:
            reason = self._build_reason(analysis)
            prompt = self._build_intervention_prompt(analysis, state)
            confidence = self._calculate_confidence(analysis)

            logger.info(
                f"Intervention decision: YES - {reason} "
                f"(confidence={confidence:.2f}, tension={analysis.tension_score:.2f})"
            )

            return InterventionDecision(
                should_intervene=True,
                reason=reason,
                suggested_prompt=prompt,
                confidence=confidence,
                cooldown_active=False
            )

        return InterventionDecision(
            should_intervene=False,
            reason=f"No intervention needed (tension={analysis.tension_score:.2f}, utterances={self._utterances_since_intervention})",
            suggested_prompt=None,
            confidence=1.0 - analysis.tension_score,
            cooldown_active=False
        )

    def record_intervention(self):
        """
        Record that an intervention occurred.
        This starts the cooldown period and resets counters.
        """
        self._last_intervention_time = time.time()
        self._utterances_since_intervention = 0
        self._consecutive_same_speaker = 0
        logger.info(f"Intervention recorded, cooldown period started ({self._cooldown_seconds}s)")

    def _build_checkin_prompt(self, analysis: AnalysisResult, state: ConversationState) -> str:
        """Build a proactive check-in prompt."""
        recent = state.get_recent_transcript(n=3)
        transcript_text = "\n".join(f"{u.speaker}: {u.text}" for u in recent) if recent else "No recent transcript"

        return f"""As an AI referee, provide a brief check-in to keep the conversation productive.

Recent discussion:
{transcript_text}

Current metrics:
- Tension level: {analysis.tension_score:.1%}
- Balance: {1.0 - analysis.balance_score:.1%} balanced

Give a brief (1-2 sentences) supportive comment that:
1. Acknowledges the discussion is progressing
2. Gently reminds about staying focused on solutions
3. Encourages continued collaboration"""

    def _build_balance_prompt(self, analysis: AnalysisResult, state: ConversationState) -> str:
        """Build a prompt to encourage balanced participation."""
        recent = state.get_recent_transcript(n=3)
        transcript_text = "\n".join(f"{u.speaker}: {u.text}" for u in recent) if recent else "No recent transcript"

        # Find who hasn't been speaking
        stats = state.get_stats()
        speakers = list(stats.get('speaker_stats', {}).keys())
        other_speaker = [s for s in speakers if s != self._last_speaker]
        other_name = other_speaker[0] if other_speaker else "the other founder"

        return f"""As an AI referee, gently encourage balanced participation.

Recent discussion (note: {self._last_speaker} has been speaking several times in a row):
{transcript_text}

Give a brief (1 sentence) comment that:
1. Politely invites {other_name} to share their perspective
2. Maintains a supportive, non-judgmental tone"""

    def _is_in_cooldown(self, current_time: float) -> bool:
        """Check if we're currently in cooldown period."""
        if self._last_intervention_time is None:
            return False

        elapsed = current_time - self._last_intervention_time
        return elapsed < self._cooldown_seconds

    def _get_cooldown_remaining(self, current_time: float) -> float:
        """Get remaining cooldown time in seconds."""
        if self._last_intervention_time is None:
            return 0.0

        elapsed = current_time - self._last_intervention_time
        remaining = max(0.0, self._cooldown_seconds - elapsed)
        return remaining

    def _build_reason(self, analysis: AnalysisResult) -> str:
        """
        Build human-readable reason for intervention.

        Args:
            analysis: Analysis result

        Returns:
            Reason string
        """
        reasons = []

        if analysis.tension_score >= 0.8:
            reasons.append("high tension")
        elif analysis.tension_score >= 0.7:
            reasons.append("elevated tension")

        if analysis.balance_score >= 0.8:
            reasons.append("severe imbalance")
        elif analysis.balance_score >= 0.6:
            reasons.append("speaking imbalance")

        if analysis.interruption_rate > 5.0:
            reasons.append("excessive interruptions")

        if analysis.dominant_speaker:
            reasons.append(f"{analysis.dominant_speaker} dominating")

        if not reasons:
            reasons.append("conversation health concern")

        return ", ".join(reasons)

    def _build_intervention_prompt(
        self,
        analysis: AnalysisResult,
        state: ConversationState
    ) -> str:
        """
        Build suggested intervention prompt for the LLM.

        Args:
            analysis: Analysis result
            state: Conversation state

        Returns:
            Prompt string for LLM
        """
        stats = state.get_stats()

        # Get recent transcript
        recent = state.get_recent_transcript(n=5)
        transcript_text = "\n".join(
            f"{u.speaker}: {u.text}" for u in recent
        )

        # Build context about detected issues
        issues = []

        if analysis.dominant_speaker:
            issues.append(
                f"{analysis.dominant_speaker} has been speaking "
                f"{stats['speaker_stats'][analysis.dominant_speaker]['speaking_percentage']:.0f}% of the time"
            )

        if analysis.interruption_rate > 3.0:
            issues.append(f"High interruption rate: {analysis.interruption_rate:.1f} per minute")

        if analysis.balance_score > 0.6:
            issues.append("Significant speaking time imbalance detected")

        for pattern in analysis.detected_patterns:
            issues.append(pattern)

        issues_text = "\n".join(f"- {issue}" for issue in issues)

        prompt = f"""As an AI referee moderating a founder conversation, you've detected the following issues:

{issues_text}

Recent conversation:
{transcript_text}

Please intervene with a brief, diplomatic message to help rebalance the conversation.
Keep your response concise (2-3 sentences) and constructive."""

        return prompt

    def _calculate_confidence(self, analysis: AnalysisResult) -> float:
        """
        Calculate confidence in intervention decision.

        Args:
            analysis: Analysis result

        Returns:
            Confidence score from 0.0 to 1.0
        """
        # Higher tension and more detected patterns = higher confidence
        pattern_factor = min(len(analysis.detected_patterns) / 3.0, 1.0)
        tension_factor = analysis.tension_score

        confidence = 0.6 * tension_factor + 0.4 * pattern_factor
        return min(confidence, 1.0)

    def set_cooldown(self, seconds: int):
        """
        Update cooldown period.

        Args:
            seconds: New cooldown period in seconds

        Raises:
            ValueError: If seconds is negative
        """
        if seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")

        logger.info(f"Cooldown period updated: {self._cooldown_seconds}s -> {seconds}s")
        self._cooldown_seconds = seconds

    def reset_cooldown(self):
        """Reset cooldown timer (allow immediate intervention)."""
        logger.info("Cooldown timer reset")
        self._last_intervention_time = None

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"InterventionDecider(cooldown={self._cooldown_seconds}s, "
            f"last_intervention={self._last_intervention_time})"
        )
