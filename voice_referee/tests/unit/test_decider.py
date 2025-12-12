"""
Unit tests for InterventionDecider module

Tests the decide() method which takes AnalysisResult and ConversationState
and returns InterventionDecision.
"""

import pytest
import time
from unittest.mock import patch

from decision.intervention_decider import InterventionDecider, InterventionDecision
from analysis.conversation_analyzer import AnalysisResult
from processors.conversation_state import ConversationState


class TestInterventionDecider:
    """Test suite for InterventionDecider class"""

    @pytest.fixture
    def decider(self):
        """Fixture providing a fresh InterventionDecider instance"""
        return InterventionDecider(cooldown_seconds=30, utterances_between_checkins=5)

    @pytest.fixture
    def state(self):
        """Fixture providing a fresh ConversationState instance"""
        return ConversationState(max_buffer_size=50)

    @pytest.fixture
    def calm_analysis(self):
        """Fixture providing calm conversation analysis"""
        return AnalysisResult(
            tension_score=0.2,
            balance_score=0.1,
            interruption_rate=0.0,
            dominant_speaker=None,
            detected_patterns=[],
            requires_intervention=False
        )

    @pytest.fixture
    def tense_analysis(self):
        """Fixture providing tense conversation analysis"""
        return AnalysisResult(
            tension_score=0.8,
            balance_score=0.6,
            interruption_rate=5.0,
            dominant_speaker="Founder A",
            detected_patterns=["High interruption frequency detected"],
            requires_intervention=True
        )

    def test_decider_initialization(self):
        """Test InterventionDecider initialization"""
        decider = InterventionDecider(cooldown_seconds=60, utterances_between_checkins=10)
        assert decider._cooldown_seconds == 60
        assert decider._utterances_between_checkins == 10
        assert decider._last_intervention_time is None

    def test_decider_initialization_default_values(self):
        """Test InterventionDecider with default values"""
        decider = InterventionDecider()
        assert decider._cooldown_seconds == 30
        assert decider._utterances_between_checkins == 5

    def test_decider_negative_cooldown_raises_error(self):
        """Test negative cooldown raises ValueError"""
        with pytest.raises(ValueError, match="must be non-negative"):
            InterventionDecider(cooldown_seconds=-1)

    def test_decide_returns_intervention_decision(self, decider, calm_analysis, state):
        """Test decide returns InterventionDecision"""
        result = decider.decide(calm_analysis, state)

        assert isinstance(result, InterventionDecision)
        assert hasattr(result, 'should_intervene')
        assert hasattr(result, 'reason')
        assert hasattr(result, 'suggested_prompt')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'cooldown_active')

    def test_decide_no_intervention_for_calm(self, decider, calm_analysis, state):
        """Test no intervention for calm conversation"""
        result = decider.decide(calm_analysis, state)

        assert result.should_intervene is False
        assert result.cooldown_active is False

    def test_decide_intervention_for_tense(self, decider, tense_analysis, state):
        """Test intervention triggered for tense conversation"""
        # Add some utterances to state
        state.add_utterance("Test", "Founder A", time.time(), 1.0)

        result = decider.decide(tense_analysis, state)

        assert result.should_intervene is True
        assert result.suggested_prompt is not None
        assert result.confidence > 0

    def test_cooldown_blocks_intervention(self, decider, tense_analysis, state):
        """Test cooldown period blocks intervention"""
        state.add_utterance("Test", "Founder A", time.time(), 1.0)

        # Record intervention to start cooldown
        decider.record_intervention()

        # Now decide again - should be blocked by cooldown
        result = decider.decide(tense_analysis, state)

        assert result.should_intervene is False
        assert result.cooldown_active is True
        assert "cooldown" in result.reason.lower()

    def test_cooldown_expires(self, decider, tense_analysis, state):
        """Test intervention allowed after cooldown expires"""
        state.add_utterance("Test", "Founder A", time.time(), 1.0)

        # Set last intervention to well in the past
        decider._last_intervention_time = time.time() - 100  # 100 seconds ago

        result = decider.decide(tense_analysis, state)

        # Should not be blocked by cooldown
        assert result.cooldown_active is False

    def test_record_intervention_starts_cooldown(self, decider):
        """Test record_intervention sets last intervention time"""
        assert decider._last_intervention_time is None

        decider.record_intervention()

        assert decider._last_intervention_time is not None
        assert decider._utterances_since_intervention == 0

    def test_proactive_checkin_after_many_utterances(self, decider, calm_analysis, state):
        """Test proactive check-in triggered after many utterances"""
        # Add utterances to state
        for i in range(6):
            state.add_utterance(f"Utterance {i}", "Founder A", time.time() + i, 1.0)
            # Simulate decide being called (increments utterance counter)
            decider.decide(calm_analysis, state)

        # After 5+ utterances, should trigger proactive check-in
        result = decider.decide(calm_analysis, state)

        # Should suggest intervention for check-in
        assert result.should_intervene is True
        assert "utterance" in result.reason.lower() or "check-in" in result.reason.lower()

    def test_consecutive_same_speaker_triggers_intervention(self, decider, calm_analysis, state):
        """Test intervention when same speaker talks too much"""
        # Same speaker talks 4 times in a row
        for i in range(4):
            state.add_utterance(f"Speech {i}", "Founder A", time.time() + i, 2.0)
            decider.decide(calm_analysis, state)

        result = decider.decide(calm_analysis, state)

        # Should suggest intervention (either for consecutive speaker or periodic check-in)
        assert result.should_intervene is True
        # The reason should mention something about engagement or balancing
        assert any(term in result.reason.lower() for term in
                   ['consecutive', 'founder a', 'check-in', 'engagement', 'utterance'])

    def test_set_cooldown(self, decider):
        """Test updating cooldown period"""
        decider.set_cooldown(120)
        assert decider._cooldown_seconds == 120

    def test_set_cooldown_negative_raises_error(self, decider):
        """Test negative cooldown update raises ValueError"""
        with pytest.raises(ValueError, match="must be non-negative"):
            decider.set_cooldown(-10)

    def test_reset_cooldown(self, decider):
        """Test reset_cooldown allows immediate intervention"""
        decider.record_intervention()
        assert decider._last_intervention_time is not None

        decider.reset_cooldown()
        assert decider._last_intervention_time is None

    def test_intervention_reason_includes_context(self, decider, tense_analysis, state):
        """Test intervention reason provides context"""
        state.add_utterance("Test", "Founder A", time.time(), 1.0)
        result = decider.decide(tense_analysis, state)

        # Reason should provide meaningful context
        assert len(result.reason) > 0
        # Should mention something about the issue
        assert any(term in result.reason.lower() for term in
                   ['tension', 'imbalance', 'interruption', 'dominating', 'concern'])

    def test_suggested_prompt_for_intervention(self, decider, tense_analysis, state):
        """Test suggested prompt is provided for interventions"""
        state.add_utterance("Test", "Founder A", time.time(), 1.0)
        result = decider.decide(tense_analysis, state)

        if result.should_intervene:
            assert result.suggested_prompt is not None
            assert len(result.suggested_prompt) > 0

    def test_confidence_score_bounds(self, decider, tense_analysis, state):
        """Test confidence score is bounded 0.0-1.0"""
        state.add_utterance("Test", "Founder A", time.time(), 1.0)
        result = decider.decide(tense_analysis, state)

        assert 0.0 <= result.confidence <= 1.0

    def test_repr(self, decider):
        """Test string representation"""
        repr_str = repr(decider)
        assert "InterventionDecider" in repr_str
        assert "cooldown" in repr_str
